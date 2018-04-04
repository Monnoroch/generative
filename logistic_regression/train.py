import argparse
import sys

import numpy as np
import tensorflow as tf

from logistic_regression import model
from common.experiment import Experiment, load_checkpoint
from datasets import gaussian_mixture


def print_graph(session, model, step):
    """
    A helper function for printing key training characteristics.
    """
    loss = session.run(model.loss)
    print("Model on step %d has loss = %f" % (step, loss))


def make_dataset(params, training_params):
    return gaussian_mixture.dataset(params).repeat().batch(training_params.batch_size).make_one_shot_iterator()


def main(args):
    """
    The main function to train the model.
    """
    parser = argparse.ArgumentParser(description="Train the gan-normal model.")
    Experiment.add_arguments(parser)
    model.TrainingParams.add_arguments(parser)
    gaussian_mixture.DatasetParams.add_arguments(parser)
    args = parser.parse_args(args)

    experiment = Experiment.from_args(args)

    # Create the model.
    training_params = model.TrainingParams(args)
    dataset = make_dataset(gaussian_mixture.DatasetParams(args), training_params)
    model_ops = model.LogisticRegressionModel(dataset, training_params)

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
        for _ in range(training_params.max_steps):
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
