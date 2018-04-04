import argparse
import sys

import tensorflow as tf

from gan_model_data import model
from common.experiment import Experiment, load_checkpoint
from common.training_loop import TrainingLoopParams, training_loop


def print_graph(session, model, step, nn_generator):
    """
    A helper function for printing key training characteristics.
    """
    if nn_generator:
        real, fake = session.run([model.average_probability_real, model.average_probability_fake])
        print("Saved model with step %d; real = %f, fake = %f" % (step, real, fake))
    else:
        real, fake, mean, stddev = session.run([model.average_probability_real, model.average_probability_fake, model.mean, model.stddev])
        print("Saved model with step %d; real = %f, fake = %f, mean = %f, stddev = %f" % (step, real, fake, mean, stddev))


def train(session, global_step, model_ops, args, hparams):
    print_graph(session, model_ops, global_step, hparams.nn_generator)
    # First, we run one step of discriminator training.
    for _ in range(max(int(args.discriminator_steps/2), 1)):
        session.run(model_ops.discriminator_train)
    # Then we run one step of generator training.
    for _ in range(args.generator_steps):
        session.run(model_ops.generator_train)
    for _ in range(int(args.discriminator_steps/2)):
        session.run(model_ops.discriminator_train)


def main(args):
    """
    The main function to train the model.
    """
    parser = argparse.ArgumentParser(description="Train the gan-normal model.")
    parser.add_argument("--batch_size", type=int, default=32, help="The size of the minibatch")
    parser.add_argument("--d_learning_rate", type=float, default=0.01, help="The discriminator learning rate")
    parser.add_argument("--g_learning_rate", type=float, default=0.02, help="The generator learning rate")
    parser.add_argument("--d_l2_reg", type=float, default=0.0005, help="The discriminator L2 regularization parameter")
    parser.add_argument("--g_l2_reg", type=float, default=0., help="The generator L2 regularization parameter")
    parser.add_argument("--input_mean", type=float, default=[], help="The mean of the input dataset", action="append")
    parser.add_argument("--input_stddev", type=float, default=[], help="The standard deviation of the input dataset", action="append")
    parser.add_argument("--dropout", type=float, default=0.5, help="The dropout rate to use in the descriminator")
    parser.add_argument("--discriminator_steps", type=int, default=1, help="The number of steps to train the descriminator on each iteration")
    parser.add_argument("--generator_steps", type=int, default=1, help="The number of steps to train the generator on each iteration")
    parser.add_argument("--nn_generator", default=False, action="store_true", help="Whether to use a neural network as a generator")
    parser.add_argument("--generator_features", default=[], action="append", type=int, help="The number of features in generators hidden layers")
    parser.add_argument("--discriminator_features", default=[], action="append", type=int, help="The number of features in discriminators hidden layers")
    Experiment.add_arguments(parser)
    TrainingLoopParams.add_arguments(parser)
    args = parser.parse_args(args)
    # Default input mean and stddev.
    if not args.input_mean:
        args.input_mean.append(15.)
    if not args.input_stddev:
        args.input_stddev.append(7.)
    if len(args.input_mean) != len(args.input_stddev):
        print("There must be the same number of input means and standard deviations.")
        sys.exit(1)

    experiment = Experiment.from_args(args)
    hparams = experiment.load_hparams(model.ModelParams, args)

    # Create the model.
    model_ops = model.GanNormalModel(hparams, model.DatasetParams(args), model.TrainingParams(args, training=True))

    training_loop(TrainingLoopParams(args), experiment, model_ops.summaries,
        lambda session, global_step: train(session, global_step, model_ops, args, hparams),
        checkpoint=load_checkpoint(args))


if __name__ == "__main__":
    main(sys.argv[1:])
