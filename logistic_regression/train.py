import argparse
import sys

from logistic_regression import model
from common.experiment import Experiment, load_checkpoint
from common.training_loop import TrainingLoopParams, training_loop
from datasets import gaussian_mixture


def print_graph(session, model, step):
    """
    A helper function for printing key training characteristics.
    """
    loss = session.run(model.loss)
    print("Model on step %d has loss = %f" % (step, loss))


def make_dataset(params, training_params):
    return gaussian_mixture.dataset(params).batch(training_params.batch_size).make_one_shot_iterator()


def train(session, global_step, model_ops):
    print_graph(session, model_ops, global_step)
    session.run(model_ops.train)


def main(args):
    """
    The main function to train the model.
    """
    parser = argparse.ArgumentParser(description="Train the gan-normal model.")
    Experiment.add_arguments(parser)
    model.TrainingParams.add_arguments(parser)
    gaussian_mixture.DatasetParams.add_arguments(parser)
    TrainingLoopParams.add_arguments(parser)
    args = parser.parse_args(args)

    experiment = Experiment.from_args(args)

    # Create the model.
    training_params = model.TrainingParams(args)
    dataset = make_dataset(gaussian_mixture.DatasetParams(args), training_params)
    model_ops = model.LogisticRegressionModel(dataset, training_params)

    training_loop(TrainingLoopParams(args), experiment, model_ops.summaries,
        lambda session, global_step: train(session, global_step, model_ops), checkpoint=load_checkpoint(args))


if __name__ == "__main__":
    main(sys.argv[1:])
