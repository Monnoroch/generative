import argparse
import os
import sys

import numpy as np
import tensorflow as tf

from linear_regression import model


train_data_template = """
discriminatorParam1 = %s;

discriminatorParam2 = %s;

Animate[
  Plot[{
    %f * x + %f,
    discriminatorParam1[[step]] * x + discriminatorParam2[[step]]
  }, {x, -7.5, 10.}, PlotRange -> {0, 1}],
  {step, Range[1, Length[discriminatorParam1]]},
  AnimationRunning -> True,
  DefaultDuration -> 25,
  Deployed -> True,
  DisplayAllSteps -> True
]
"""

def format_list(values):
    if type(values) is not list:
        return "%f" % values
    return "{%s}" % ",".join(map(format_list, values))

def format_train_data(model_data):
    return train_data_template % (
        format_list(model_data["discriminator"]["param1"]),
        format_list(model_data["discriminator"]["param2"]),
        model_data["data"]["param1"], model_data["data"]["param2"])


def print_graph(session, model, step, model_data):
    """
    A helper function for printing key training characteristics.
    """
    loss = session.run(model.loss)
    print("Model on step %d has loss = %f" % (step, loss))

    param1, param2 = session.run([model.param1, model.param2])
    model_data["discriminator"]["param1"].append(param1)
    model_data["discriminator"]["param2"].append(param2)


def main(args):
    """
    The main function to train the model.
    """
    parser = argparse.ArgumentParser(description="Train the gan-normal model.")
    parser.add_argument("--experiment_dir", required=True, help="The expriment directory to store all the data")
    parser.add_argument("--load_checkpoint", help="Continue training from a checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="The size of the minibatch")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="The learning rate")
    parser.add_argument("--l2_reg", type=float, default=0.0005, help="The L2 regularization parameter")
    parser.add_argument("--input_param1", type=float, help="The mean of the input dataset")
    parser.add_argument("--input_param2", type=float, help="The standard deviation of the input dataset")
    parser.add_argument("--max_steps", type=int, default=2000, help="The maximum number of steps to train training for")
    args = parser.parse_args(args)

    # Initialize experiment files.
    train_dir = os.path.join(args.experiment_dir, "model")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    summaries_dir = os.path.join(args.experiment_dir, "summaries")

    # Create the model.
    tparams = model.TrainingParams(args, training=True)
    model_ops = model.LinearRegressionModel(model.DatasetParams(args), tparams)

    model_data = {
        "data": {
            "param1": args.input_param1,
            "param2": args.input_param2,
        },
        "discriminator": {
            "param1": [],
            "param2": [],
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
            print_graph(session, model_ops, global_step, model_data)
            session.run(model_ops.train)

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
