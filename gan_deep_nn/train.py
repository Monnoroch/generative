import argparse
import os
import sys

import tensorflow as tf

from gan_deep_nn import model
from common.experiment import Experiment, load_checkpoint
from common.training_loop import TrainingLoopParams, training_loop
from datasets.mnist import mnist_dataset


def print_graph(session, model, step):
    """
    A helper function for printing key training characteristics.
    """
    real, fake = session.run([model.average_probability_real, model.average_probability_fake])
    print("Step %d; real = %f, fake = %f" % (step, real, fake))


def make_dataset(args):
    train_dataset = mnist_dataset(args.dataset_dir, train=True)
    test_dataset = mnist_dataset(args.dataset_dir, train=False)
    dataset = train_dataset.concatenate(test_dataset)
    dataset = dataset.map(lambda image, label: image) # Only images.
    dataset = dataset.map(lambda image: tf.reshape(image, [28, 28, 1]))
    dataset = dataset.repeat()
    dataset = dataset.batch(args.batch_size)
    return dataset.make_one_shot_iterator()


def train(session, global_step, model_ops, args):
    print_graph(session, model_ops, global_step)
    # First, we run one step of discriminator training.
    for _ in range(args.discriminator_steps):
        session.run(model_ops.discriminator_train)

    # Then we run one step of generator training.
    for _ in range(args.generator_steps):
        session.run(model_ops.generator_train)


def main(args):
    """
    The main function to train the model.
    """
    parser = argparse.ArgumentParser(description="Train the gan-normal model.")
    parser.add_argument("--dataset_dir", required=True, help="The path to the MNIST dataset files. If doesn't exist, the dataset will be downloaded")
    parser.add_argument("--batch_size", type=int, default=32, help="The size of the minibatch")
    parser.add_argument("--d_learning_rate", type=float, default=0.005, help="The discriminator learning rate")
    parser.add_argument("--g_learning_rate", type=float, default=0.005, help="The generator learning rate")
    parser.add_argument("--d_l2_reg", type=float, default=0., help="The discriminator L2 regularization parameter")
    parser.add_argument("--g_l2_reg", type=float, default=0., help="The generator L2 regularization parameter")
    parser.add_argument("--dropout", type=float, default=0.5, help="The dropout rate to use in the descriminator")
    parser.add_argument("--discriminator_steps", type=int, default=1, help="The number of steps to train the descriminator on each iteration")
    parser.add_argument("--generator_steps", type=int, default=1, help="The number of steps to train the generator on each iteration")
    parser.add_argument("--generator_features", default=[], action="append", type=int, help="The number of features in generators hidden layers")
    parser.add_argument("--discriminator_features", default=[], action="append", type=int, help="The number of features in discriminators hidden layers")
    parser.add_argument("--latent_space_size", default=128, type=int, help="The number of features in generator input")
    parser.add_argument("--smooth_labels", default=False, action="store_true", help="Whether to use smooth or sharp labels")
    Experiment.add_arguments(parser)
    TrainingLoopParams.add_arguments(parser)
    args = parser.parse_args(args)

    experiment = Experiment.from_args(args)
    hparams = experiment.load_hparams(model.ModelParams, args)

    dataset = make_dataset(args)
    model_ops = model.GanModel(hparams, dataset, model.TrainingParams(args, training=True))

    training_loop(TrainingLoopParams(args), experiment, model_ops.summaries,
        lambda session, global_step: train(session, global_step, model_ops, args),
        checkpoint=load_checkpoint(args))


if __name__ == "__main__":
    main(sys.argv[1:])
