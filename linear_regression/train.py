import argparse
import sys

import numpy as np
import tensorflow as tf

from linear_regression import model, dataset
from common.experiment import Experiment, load_checkpoint
from common.training_loop import TrainingLoopParams, training_loop


def print_graph(session, model, step):
    """
    A helper function for printing key training characteristics.
    """
    loss = session.run(model.loss)
    print("Model on step %d has loss = %f" % (step, loss))


def make_dataset(params):
    return dataset.linear_dependent_with_error(params).make_one_shot_iterator()


def train(session, global_step, model_ops):
    print_graph(session, model_ops, global_step)
    session.run(model_ops.train)


def main(args):
    """
    The main function to train the model.
    """
    parser = argparse.ArgumentParser(description="Train the gan-normal model.")
    parser.add_argument("--batch_size", type=int, default=32, help="The size of the minibatch")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="The learning rate")
    parser.add_argument("--l2_reg", type=float, default=0.0005, help="The L2 regularization parameter")
    parser.add_argument("--input_param1", type=float, default=[], help="The mean of the input dataset", action="append")
    parser.add_argument("--input_param2", type=float, default=[], help="The standard deviation of the input dataset", action="append")
    Experiment.add_arguments(parser)
    TrainingLoopParams.add_arguments(parser)
    args = parser.parse_args(args)

    experiment = Experiment.from_args(args)

    # Create the model.
    dataset_value = make_dataset(dataset.DatasetParams(args))
    model_ops = model.LinearRegressionModel(dataset_value, model.TrainingParams(args))

    training_loop(TrainingLoopParams(args), experiment, model_ops.summaries,
        lambda session, global_step: train(session, global_step, model_ops), checkpoint=load_checkpoint(args))


if __name__ == "__main__":
    main(sys.argv[1:])
