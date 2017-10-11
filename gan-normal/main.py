import argparse
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
        self.dropout = args.dropout
        self.input_mean = args.input_mean
        self.input_stddev = args.input_stddev


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
            self.generated = self.generator(self.generator_input)

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

        # Optimize losses with Adam optimizer.
        self.generator_train = tf.train.AdamOptimizer(hparams.g_learning_rate).minimize(
            self.generator_loss, var_list=generator_variables, name="train_generator")
        self.discriminator_train = tf.train.AdamOptimizer(hparams.d_learning_rate).minimize(
            self.discriminator_loss, var_list=discriminator_variables, name="train_discriminator")

        # Add useful graphs to Tensorboard.
        self.average_probability_real = tf.reduce_mean(tf.sigmoid(self.real_ratings))
        self.average_probability_fake = tf.reduce_mean(tf.sigmoid(self.generated_ratings))

        tf.summary.scalar("D/cost", self.discriminator_loss)
        tf.summary.scalar("G/cost", self.generator_loss)
        tf.summary.scalar("P_real_on_real/", self.average_probability_real)
        tf.summary.scalar("P_real_on_fake/", self.average_probability_fake)
        tf.summary.scalar("G/mean", self.mean)
        tf.summary.scalar("G/stddev", self.stddev)
        self.summaries = tf.summary.merge_all()

    def discriminator(self, input, hparams):
        """
        Discriminator is a theee-layer fully connected NN.
        """
        # First fully connected layer with 256 features and optional dropout.
        input_size = 1
        features = 256
        weights = tf.get_variable("weights_1", initializer=tf.truncated_normal([input_size, features], stddev=0.1))
        biases = tf.get_variable("biases_1", initializer=tf.constant(0.1, shape=[features]))
        hidden_layer = tf.nn.relu(tf.matmul(input, weights) + biases)
        if hparams.dropout != 0.0:
            hidden_layer = tf.nn.dropout(hidden_layer, hparams.dropout)

        # Second fully connected layer with 256 features and optional dropout.
        features = 256
        weights = tf.get_variable("weights_2", initializer=tf.truncated_normal([input_size, features], stddev=0.1))
        biases = tf.get_variable("biases_2", initializer=tf.constant(0.1, shape=[features]))
        hidden_layer = tf.nn.relu(tf.matmul(input, weights) + biases)
        if hparams.dropout != 0.0:
            hidden_layer = tf.nn.dropout(hidden_layer, hparams.dropout)

        # Final linear layer to compute the classifier's logits.
        classes = 1
        weights = tf.get_variable("weights_out", initializer=tf.truncated_normal([features, classes], stddev=0.1))
        biases = tf.get_variable("biases_out", initializer=tf.constant(0.1, shape=[classes]))
        return tf.matmul(hidden_layer, weights) + biases

    def generator(self, input):
        """
        Generator is just a linear transformation of the input.
        """
        self.mean = tf.Variable(tf.constant(0.))
        # Standard deviation has to be positive, so we make sure it is by computing the absolute value of the variable.
        self.stddev = tf.sqrt(tf.Variable(tf.constant(1.)) ** 2)
        return input * self.stddev + self.mean

    def data_batch(self, hparams):
        """
        Input data are just samples from N(mean, stddev).
        """
        return tf.random_normal([hparams.batch_size, 1], hparams.input_mean, hparams.input_stddev)

    def generator_input(self, hparams):
        """
        Generator input data are just samples from N(0, 1).
        """
        return tf.random_normal([hparams.batch_size, 1], 0., 1.)

def print_graph(session, model, step):
    """
    A helper function for printing key training characteristics.
    """
    real, fake, mean, stddev = session.run([
        model.average_probability_real, model.average_probability_fake, model.mean, model.stddev
    ])
    step += 1

    print("Saved model with step %d; real = %f, fake = %f, mean = %f, stddev = %f" % (
          step, real, fake, mean, stddev))


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
    parser.add_argument("--g_learning_rate", type=float, default=0.1, help="The generator learning rate")
    parser.add_argument("--d_l2_reg", type=float, default=0.0005, help="The discriminator L2 regularization parameter")
    parser.add_argument("--input_mean", type=float, default=15., help="The mean of the input dataset")
    parser.add_argument("--input_stddev", type=float, default=7., help="The standard deviation of the input dataset")
    parser.add_argument("--max_steps", type=int, default=2000, help="The maximum number of steps to train training for")
    parser.add_argument("--dropout", type=float, default=0.5, help="The dropout rate to use in the descriminator")
    args = parser.parse_args(args)

    # Create the model.
    model = GanNormalModel(ModelParams(args))

    saver = tf.train.Saver()
    with tf.Session() as session:
        # Initializing the model. Either using a saved checkpoint or a ranrom initializer.
        if args.load_checkpoint:
            saver.restore(session, args.load_checkpoint)
        else:
            session.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(args.summaries_dir + '/train', session.graph)

        # The main training loop. On each interation we train both the discriminator and the generator on one minibatch.
        for step in range(args.max_steps):
            print_graph(session, model, step)
            # First, we run one step of discriminator training.
            session.run(model.discriminator_train)
            # Then we run one step of generator training.
            session.run(model.generator_train)
            # And export all summaries to tensorboard.
            summary_writer.add_summary(session.run(model.summaries), step)
            # saver.save(session, "%s/model.ckpt-%d" % (args.train_dir, step))


if __name__ == "__main__":
    main(sys.argv[1:])
