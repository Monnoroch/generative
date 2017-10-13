import argparse
import os
import sys

import tensorflow as tf


def add_n(arr):
  if not arr:
    return 0
  return tf.add_n(arr)


class ModelParams(object):
    """
    All model hyperparameters that should be configured from command line should go here.
    """
    def __init__(self, args):
        self.d_learning_rate = args.d_learning_rate
        self.g_learning_rate = args.g_learning_rate
        self.batch_size = args.batch_size
        self.d_l2_reg = args.d_l2_reg
        self.g_l2_reg = args.g_l2_reg
        self.dropout = args.dropout
        self.input_mean = args.input_mean
        self.input_stddev = args.input_stddev
        self.nn_generator = args.nn_generator
        self.generator_features = args.generator_features
        self.discriminator_features = args.discriminator_features


class GanNormalModel(object):
    """
    A GAN that fits a one-dimentional input data that has a normal distribution.

    The input space X is just R.
    The input data probability distribution is N(mean, stddev).
    The generator knows that the data is normally distributed and tries to find the parameters of the normal
        distribution, so it's not a neural network but rather just a linear mapping. The generator input is a random
        variable distributed as N(0, 1).
    The discriminator however is a deep fully connected NN. Dropout and weight decay are uesd to reguralize the model.
    """
    def __init__(self, hparams):
        # Get the real and the fake inputs.
        self.real_input = self.data_batch(hparams)
        self.generator_input = self.generator_input(hparams)
        with tf.variable_scope("generator"):
            self.generated = self.generator(self.generator_input, hparams)

        # Get discriminator logits for both inputs.
        with tf.variable_scope("discriminator"):
            self.real_ratings = self.discriminator(self.real_input, hparams)
        # We want to share parameters between discriminator instances because logically there's only one discriminator.
        with tf.variable_scope("discriminator", reuse=True):
            self.generated_ratings = self.discriminator(self.generated, hparams)

        # Discriminator loss minimizes the discrimination error on both real and fake inputs.
        self.loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(self.real_ratings), logits=self.real_ratings))
        self.loss_generated = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(self.generated_ratings), logits=self.generated_ratings))
        self.discriminator_loss = self.loss_generated + self.loss_real

        # Generator loss maximizes the discrimination error on fake inputs.
        self.generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(self.generated_ratings), logits=self.generated_ratings))

        with tf.variable_scope("generator") as scope:
            generator_variables = [v for v in tf.global_variables() if v.name.startswith(scope.name)]
        with tf.variable_scope("discriminator") as scope:
            discriminator_variables = [v for v in tf.global_variables() if v.name.startswith(scope.name)]
        # Optionally add L2 regularization to the discriminator.
        if hparams.d_l2_reg != 0.0:
            self.discriminator_loss += hparams.d_l2_reg * add_n([tf.nn.l2_loss(v) for v in discriminator_variables])
        if hparams.g_l2_reg != 0.0:
            self.generator_loss += hparams.g_l2_reg * add_n([tf.nn.l2_loss(v) for v in generator_variables])

        # Optimize losses with Adam optimizer.
        self.generator_train = tf.train.AdamOptimizer(hparams.g_learning_rate).minimize(
            self.generator_loss, var_list=generator_variables, name="train_generator")
        self.discriminator_train = tf.train.AdamOptimizer(hparams.d_learning_rate).minimize(
            self.discriminator_loss, var_list=discriminator_variables, name="train_discriminator")

        # Set up the global step.
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.increment_global_step = tf.assign_add(self.global_step, 1)

        # Add useful graphs to Tensorboard.
        self.average_probability_real = tf.reduce_mean(tf.sigmoid(self.real_ratings))
        self.average_probability_fake = tf.reduce_mean(tf.sigmoid(self.generated_ratings))

        tf.summary.scalar("D/cost", self.discriminator_loss)
        tf.summary.scalar("G/cost", self.generator_loss)
        tf.summary.scalar("P_real_on_real/", self.average_probability_real)
        tf.summary.scalar("P_real_on_fake/", self.average_probability_fake)
        if not hparams.nn_generator:
            tf.summary.scalar("G/mean", self.mean)
            tf.summary.scalar("G/stddev", self.stddev)
        tf.summary.histogram("Real data", self.real_input)
        tf.summary.histogram("Fake data", self.generated)
        self.summaries = tf.summary.merge_all()

    def discriminator(self, input, hparams):
        """
        Discriminator is a theee-layer fully connected NN.
        """
        # First fully connected layer with N features and optional dropout.
        input_size = 1
        features = hparams.discriminator_features
        weights = tf.get_variable("weights_1", initializer=tf.truncated_normal([input_size, features], stddev=0.1))
        biases = tf.get_variable("biases_1", initializer=tf.constant(0.1, shape=[features]))
        hidden_layer = tf.nn.relu(tf.matmul(input, weights) + biases)
        if hparams.dropout != 0.0:
            hidden_layer = tf.nn.dropout(hidden_layer, hparams.dropout)

        # Second fully connected layer with N features and optional dropout.
        features = hparams.discriminator_features
        weights = tf.get_variable("weights_2", initializer=tf.truncated_normal([input_size, features], stddev=0.1))
        biases = tf.get_variable("biases_2", initializer=tf.constant(0.1, shape=[features]))
        hidden_layer = tf.nn.relu(tf.matmul(input, weights) + biases)
        if hparams.dropout != 0.0:
            hidden_layer = tf.nn.dropout(hidden_layer, hparams.dropout)

        # Final linear layer to compute the classifier's logits.
        output_size = 1
        weights = tf.get_variable("weights_out", initializer=tf.truncated_normal([features, output_size], stddev=0.1))
        biases = tf.get_variable("biases_out", initializer=tf.constant(0.1, shape=[output_size]))
        return tf.matmul(hidden_layer, weights) + biases

    def generator(self, input, hparams):
        """
        Generator is just a linear transformation of the input.
        """
        if not hparams.nn_generator:
            self.mean = tf.Variable(tf.constant(0.))
            # Standard deviation has to be positive, so we make sure it is by computing the absolute value of the variable.
            self.stddev = tf.sqrt(tf.Variable(tf.constant(1.)) ** 2)
            return input * self.stddev + self.mean

        # First fully connected layer with N features.
        input_size = 1
        features = hparams.generator_features
        weights = tf.get_variable("weights_1", initializer=tf.truncated_normal([input_size, features], stddev=0.1))
        biases = tf.get_variable("biases_1", initializer=tf.constant(0.1, shape=[features]))
        hidden_layer = tf.nn.relu(tf.matmul(input, weights) + biases)

        # Second fully connected layer with N features.
        features = hparams.generator_features
        weights = tf.get_variable("weights_2", initializer=tf.truncated_normal([input_size, features], stddev=0.1))
        biases = tf.get_variable("biases_2", initializer=tf.constant(0.1, shape=[features]))
        hidden_layer = tf.nn.relu(tf.matmul(input, weights) + biases)

        # Final linear layer to generate the example.
        output_size = 1
        weights = tf.get_variable("weights_out", initializer=tf.truncated_normal([features, output_size], stddev=0.1))
        biases = tf.get_variable("biases_out", initializer=tf.constant(0.1, shape=[output_size]))
        return tf.matmul(hidden_layer, weights) + biases

    def data_batch(self, hparams):
        """
        Input data are just samples from N(mean, stddev).
        """
        count = len(hparams.input_mean)
        componens = []
        for i in range(count):
            componens.append(tf.contrib.distributions.Normal(loc=hparams.input_mean[i], scale=hparams.input_stddev[i]))
        return tf.contrib.distributions.Mixture(
          cat=tf.contrib.distributions.Categorical(probs=[1./count] * count),
          components=componens).sample(sample_shape=[hparams.batch_size, 1])

    def generator_input(self, hparams):
        """
        Generator input data are just samples from N(0, 1).
        """
        return tf.random_normal([hparams.batch_size, 1], 0., 1.)

def print_graph(session, model, step):
    """
    A helper function for printing key training characteristics.
    """
    real, fake = session.run([model.average_probability_real, model.average_probability_fake])
    print("Saved model with step %d; real = %f, fake = %f" % (step, real, fake))


def main(args):
    """
    The main function to train the model.
    """
    parser = argparse.ArgumentParser(description="Train the gan-normal model.")
    parser.add_argument("--train_dir", required=True, help="The training directory to store all the data")
    parser.add_argument("--summaries_dir", required=True, help="The directory to save summaries to")
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

    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)

    # Create the model.
    model = GanNormalModel(ModelParams(args))

    saver = tf.train.Saver()
    with tf.Session() as session:
        # Initializing the model. Either using a saved checkpoint or a ranrom initializer.
        if args.load_checkpoint:
            saver.restore(session, args.load_checkpoint)
        else:
            session.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(args.summaries_dir + "/train", session.graph)

        # The main training loop. On each interation we train both the discriminator and the generator on one minibatch.
        global_step = session.run(model.global_step)
        for _ in range(args.max_steps):
            print_graph(session, model, global_step)
            # First, we run one step of discriminator training.
            for _ in range(max(int(args.discriminator_steps/2), 1)):
                session.run(model.discriminator_train)
            # Then we run one step of generator training.
            for _ in range(args.generator_steps):
                session.run(model.generator_train)
            for _ in range(int(args.discriminator_steps/2)):
                session.run(model.discriminator_train)

            # Increment global step.
            session.run(model.increment_global_step)
            global_step = session.run(model.global_step)
            # And export all summaries to tensorboard.
            summary_writer.add_summary(session.run(model.summaries), global_step)

        saver.save(session, "%s/model.ckpt-%d" % (args.train_dir, global_step))


if __name__ == "__main__":
    main(sys.argv[1:])
