import argparse
import json
import os
import sys
import math

import numpy as np
import tensorflow as tf

from gan_model import model
from common.hparams import load_hparams, save_hparams


def find_min(values):
    min_index = 0
    min_value = values[min_index]
    for i in range(len(values)):
        if values[i] < min_value:
            min_value = values[i]
            min_index = i
    return min_value, min_index

def find_max(values):
    max_index = 0
    max_value = values[max_index]
    for i in range(len(values)):
        if values[i] > max_value:
            max_value = values[i]
            max_index = i
    return max_value, max_index

train_data_template = """
discriminatorStepData = %s;

generatorStepData = %s;

Animate[
  Show[
    Plot[{
      0.5,
      PDF[MixtureDistribution[%s, %s], x]
    }, {x, -7.5, 10}, PlotRange -> {0, 1}],
    ListLinePlot[
      generatorStepData[[step]],
      InterpolationOrder -> 1
    ]
    ListLinePlot[
      discriminatorStepData[[step]],
      InterpolationOrder -> 3
    ]
  ],
  {step, Range[1, Length[discriminatorStepData]]},
  AnimationRunning -> True,
  DefaultDuration -> 25,
  Deployed -> True,
  DisplayAllSteps -> True
]
"""

def mixture_distribution_weights(means):
    return "{%s}" % ",".join(map(str, [0.5] * len(means)))

def mixture_distribution_modes(means, stddevs):
    return "{%s}" % ",".join(map(lambda i: "NormalDistribution[%f, %f]" % (means[i], stddevs[i]), range(len(means))))

def format_list(values):
    if type(values) is not list:
        return "%f" % values
    return "{%s}" % ",".join(map(format_list, values))

def format_long_list(values):
    if type(values) is not list:
        return format_list(values)
    return "{\n  %s\n}" % ",\n  ".join(map(format_list, values))


def format_train_data(model_data):
    return train_data_template % (
        format_long_list(model_data["discriminator"]["points"]),
        format_long_list(model_data["generator"]["points"]),
        mixture_distribution_weights(model_data["data"]["means"]),
        mixture_distribution_modes(model_data["data"]["means"], model_data["data"]["stddevs"]))

def print_graph(session, model, step, model_data, tparams):
    """
    A helper function for printing key training characteristics.
    """
    real, fake = session.run([model.average_probability_real, model.average_probability_fake])
    print("Saved model with step %d; real = %f, fake = %f" % (step, real, fake))

    batch_size = tparams.batch_size
    interval_begin = -7.5
    interval_end = 10.
    interval_step = 0.1
    values = np.arange(interval_begin, interval_end, interval_step)
    values_count = len(values)
    values = np.reshape(np.concatenate((values, np.repeat(0., batch_size - values_count))), (batch_size, 1))
    discriminator_values = session.run(model.probs, feed_dict={model.real_input: values})
    points = []
    for i in range(values_count):
        points.append([values[i][0], discriminator_values[i][0]])
    model_data["discriminator"]["points"].append(points)

    # Treat values as intervals. (value[i], value[i+1]).
    # For every interval, compute generator PDF value.
    # Output pointe ((value[i+1] - value[i])/2, PDF[i]).

    interval_step  = 0.25
    values = np.arange(interval_begin, interval_end, interval_step)
    pdf = np.zeros((len(values) - 1), dtype=int)
    sum_pdf = np.sum(pdf)
    while sum_pdf < batch_size * 1000:
        samples = session.run(model.generated)
        intervals = np.floor((np.reshape(samples, (batch_size,)) - interval_begin) / interval_step).astype(int)
        intervals = intervals[(intervals > 0) & (intervals < len(pdf))]
        unique, counts = np.unique(intervals, return_counts=True)
        pdf[unique] += counts
        sum_pdf = np.sum(pdf)

    generator_points = []
    pdf = (pdf * 1.) / np.sum(pdf)
    for i in range(len(pdf)):
        begin = values[i]
        end = values[i + 1]
        middle = (end - begin) / 2
        generator_points.append([middle, pdf[i]])
    model_data["generator"]["points"].append(generator_points)


def main(args):
    """
    The main function to train the model.
    """
    parser = argparse.ArgumentParser(description="Train the gan-normal model.")
    parser.add_argument("--experiment_dir", required=True, help="The expriment directory to store all the data")
    parser.add_argument("--load_checkpoint", help="Continue training from a checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="The size of the minibatch")
    parser.add_argument("--d_learning_rate", type=float, default=0.01, help="The discriminator learning rate")
    parser.add_argument("--g_learning_rate", type=float, default=0.02, help="The generator learning rate")
    parser.add_argument("--d_l2_reg", type=float, default=0.0005, help="The discriminator L2 regularization parameter")
    parser.add_argument("--g_l2_reg", type=float, default=0., help="The generator L2 regularization parameter")
    parser.add_argument("--input_mean", type=float, default=[], help="The mean of the input dataset", action="append")
    parser.add_argument("--input_stddev", type=float, default=[], help="The standard deviation of the input dataset", action="append")
    parser.add_argument("--max_steps", type=int, default=2000, help="The maximum number of steps to train training for")
    parser.add_argument("--dropout", type=float, default=0.5, help="The dropout rate to use in the descriminator")
    parser.add_argument("--discriminator_steps", type=int, default=1, help="The number of steps to train the descriminator on each iteration")
    parser.add_argument("--generator_steps", type=int, default=1, help="The number of steps to train the generator on each iteration")
    parser.add_argument("--nn_generator", default=False, action="store_true", help="Whether to use a neural network as a generator")
    parser.add_argument("--generator_features", default=256, type=int, help="The number of features in generators hidden layers")
    parser.add_argument("--discriminator_features", default=256, type=int, help="The number of features in discriminators hidden layers")
    args = parser.parse_args(args)
    # Default input mean and stddev.
    if not args.input_mean:
        args.input_mean.append(15.)
    if not args.input_stddev:
        args.input_stddev.append(7.)
    if len(args.input_mean) != len(args.input_stddev):
        print("There must be the same number of input means and standard deviations.")
        sys.exit(1)

    # Initialize experiment files.
    train_dir = os.path.join(args.experiment_dir, "model")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    summaries_dir = os.path.join(args.experiment_dir, "summaries")
    hparams = model.ModelParams(load_hparams(args))
    save_hparams(hparams, args)

    # Create the model.
    tparams = model.TrainingParams(args, training=True)
    model_ops = model.GanNormalModel(hparams, model.DatasetParams(args), tparams)

    model_data = {
        "data": {
            "means": args.input_mean,
            "stddevs": args.input_stddev,
        },
        "discriminator": {
            "points": [],
        },
        "generator": {
            "points": [],
        },
    }

    saver = tf.train.Saver()
    with tf.Session() as session:
        # Initializing the model. Either using a saved checkpoint or a ranrom initializer.
        if args.load_checkpoint:
            saver.restore(session, args.load_checkpoint)
        else:
            session.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(summaries_dir, session.graph)

        # The main training loop. On each interation we train both the discriminator and the generator on one minibatch.
        global_step = session.run(model_ops.global_step)
        for _ in range(args.max_steps):
            print_graph(session, model_ops, global_step, model_data, tparams)
            # First, we run one step of discriminator training.
            for _ in range(max(int(args.discriminator_steps/2), 1)):
                session.run(model_ops.discriminator_train)
            # Then we run one step of generator training.
            for _ in range(args.generator_steps):
                session.run(model_ops.generator_train)
            for _ in range(int(args.discriminator_steps/2)):
                session.run(model_ops.discriminator_train)

            # Increment global step.
            session.run(model_ops.increment_global_step)
            global_step = session.run(model_ops.global_step)
            # And export all summaries to tensorboard.
            summary_writer.add_summary(session.run(model_ops.summaries), global_step)

        # Save experiment data.
        saver.save(session, os.path.join(train_dir, "checkpoint-%d" % global_step, "data"))

    with open(os.path.join(train_dir, "train-data.mtmt.txt"), "w") as file:
        file.write(format_train_data(model_data))
        file.write("\n")


if __name__ == "__main__":
    main(sys.argv[1:])
