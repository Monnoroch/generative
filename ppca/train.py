import argparse
import sys

import numpy as np
import tensorflow as tf

from ppca import model, dataset
from common.experiment import Experiment, load_checkpoint


def print_graph(session, model, step):
    """
    A helper function for printing key training characteristics.
    """
    loss = session.run(model.loss)
    print("Model on step %d has loss = %f" % (step, loss))


def make_dataset(params):
    return dataset.normal_samples(params).make_one_shot_iterator()


def main(args):
    """
    The main function to train the model.
    """
    parser = argparse.ArgumentParser(description="Train the gan-normal model.")
    parser.add_argument("--batch_size", type=int, default=32, help="The size of the minibatch")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="The learning rate")
    parser.add_argument("--l2_reg", type=float, default=0.0005, help="The L2 regularization parameter")
    parser.add_argument("--latent_space_size", type=int, default=2, help="The latent space size")
    parser.add_argument("--input_mean", type=float, default=[], help="The mean of the input dataset", action="append")
    parser.add_argument("--input_stddev", type=float, default=[], help="The standard deviation of the input dataset", action="append")
    parser.add_argument("--max_steps", type=int, default=2000, help="The maximum number of steps to train training for")
    Experiment.add_arguments(parser)
    args = parser.parse_args(args)
    if len(args.input_mean) != len(args.input_stddev):
        print("There must be the same number of input means and standard deviations.")
        sys.exit(1)

    experiment = Experiment.from_args(args)
    hparams = experiment.load_hparams(model.ModelParams, args)

    # Create the model.
    dataset_value = make_dataset(dataset.DatasetParams(args))
    model_ops = model.PpcaModel(dataset_value, hparams, model.TrainingParams(args), args.batch_size)

    saver = tf.train.Saver()
    with tf.Session() as session:
        # Initializing the model. Either using a saved checkpoint or a ranrom initializer.
        checkpoint = load_checkpoint(args)
        if checkpoint:
            saver.restore(session, checkpoint)
        else:
            session.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(experiment.summaries_dir(), session.graph)

        # The main training loop. On each interation we train the model on one minibatch.
        global_step = session.run(model_ops.global_step)
        for _ in range(args.max_steps):
            print_graph(session, model_ops, global_step)
            session.run(model_ops.train)

            # Increment global step.
            session.run(model_ops.increment_global_step)
            global_step = session.run(model_ops.global_step)
            # And export all summaries to tensorboard.
            summary_writer.add_summary(session.run(model_ops.summaries), global_step)

        # Save experiment data.
        saver.save(session, experiment.checkpoint(global_step))


if __name__ == "__main__":
    main(sys.argv[1:])
