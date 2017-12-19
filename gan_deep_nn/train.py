import argparse
import os
import sys

import tensorflow as tf

from gan_deep_nn import model
from common.experiment import Experiment
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
    dataset = dataset.map(lambda image: tf.reshape(image, [14*2, 14*2, 1]))
    if args.normalized_input:
        dataset = dataset.map(lambda image: (image - 0.5) * 2)
    dataset = dataset.repeat()
    dataset = dataset.batch(args.batch_size)
    return dataset.make_one_shot_iterator()


def main(args):
    """
    The main function to train the model.
    """
    parser = argparse.ArgumentParser(description="Train the gan-normal model.")
    parser.add_argument("--dataset_dir", required=True, help="The path to the MNIST dataset files. If doesn't exist, the dataset will be downloaded")
    parser.add_argument("--experiment_dir", required=True, help="The expriment directory to store all the data")
    parser.add_argument("--load_checkpoint", help="Continue training from a checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="The size of the minibatch")
    parser.add_argument("--d_learning_rate", type=float, default=0.005, help="The discriminator learning rate")
    parser.add_argument("--g_learning_rate", type=float, default=0.005, help="The generator learning rate")
    parser.add_argument("--d_l2_reg", type=float, default=0., help="The discriminator L2 regularization parameter")
    parser.add_argument("--g_l2_reg", type=float, default=0., help="The generator L2 regularization parameter")
    parser.add_argument("--max_steps", type=int, default=2000, help="The maximum number of steps to train training for")
    parser.add_argument("--dropout", type=float, default=0.5, help="The dropout rate to use in the descriminator")
    parser.add_argument("--gen_dropout", type=float, default=0.0, help="The dropout rate to use in the generator")
    parser.add_argument("--discriminator_steps", type=int, default=1, help="The number of steps to train the descriminator on each iteration")
    parser.add_argument("--generator_steps", type=int, default=1, help="The number of steps to train the generator on each iteration")
    parser.add_argument("--generator_features", default=[], action="append", type=int, help="The number of features in generators hidden layers")
    parser.add_argument("--discriminator_features", default=[], action="append", type=int, help="The number of features in discriminators hidden layers")
    parser.add_argument("--latent_space_size", default=128, type=int, help="The number of features in generator input")
    parser.add_argument("--optimizer", default="adam", type=str, help="The optimizer to use for training the discriminator")
    parser.add_argument("--gen_optimizer", default="adam", type=str, help="The optimizer to use for training the generator")
    parser.add_argument("--use_batch_norm", default=False, action="store_true", help="Whether to use batch normalization in the discriminator")
    parser.add_argument("--use_gen_batch_norm", default=False, action="store_true", help="Whether to use batch normalization in the generator")
    parser.add_argument("--normalized_input", default=False, action="store_true", help="Whether to normalize inputs to [-1, 1]")
    parser.add_argument("--use_leaky_relus", default=False, action="store_true", help="Whether to use leaky relus")
    parser.add_argument("--smooth_labels", default=False, action="store_true", help="Whether to use smooth or sharp labels")
    args = parser.parse_args(args)

    experiment = Experiment(args.experiment_dir)
    hparams = experiment.load_hparams(model.ModelParams, args)

    dataset = make_dataset(args)
    model_ops = model.GanModel(hparams, dataset, model.TrainingParams(args, training=True))

    stop_generator_op = (model_ops.average_probability_fake > model_ops.average_probability_real)

    saver = tf.train.Saver()
    with tf.Session() as session:
        # Initializing the model. Either using a saved checkpoint or a ranrom initializer.
        if args.load_checkpoint:
            saver.restore(session, args.load_checkpoint)
        else:
            session.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(experiment.summaries_dir(), session.graph)

        # The main training loop. On each interation we train both the discriminator and the generator on one minibatch.
        global_step = session.run(model_ops.global_step)
        for _ in range(args.max_steps):
            print_graph(session, model_ops, global_step)
            # First, we run one step of discriminator training.
            for _ in range(args.discriminator_steps):
                session.run(model_ops.discriminator_train)

            # Then we run one step of generator training.
            for _ in range(args.generator_steps):
                session.run(model_ops.generator_train)

            # Increment global step.
            session.run(model_ops.increment_global_step)
            global_step = session.run(model_ops.global_step)
            # And export all summaries to tensorboard.
            if global_step % 10 == 0:
                summary_writer.add_summary(session.run(model_ops.summaries), global_step)
            if global_step % 5000 == 0:
                saver.save(session, experiment.checkpoint(global_step))

        # Save experiment data.
        saver.save(session, experiment.checkpoint(global_step))


if __name__ == "__main__":
    main(sys.argv[1:])
