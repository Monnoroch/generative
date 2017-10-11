import argparse
import sys

import tensorflow as tf


def add_n(arr):
  if not arr:
    return 0
  return tf.add_n(arr)


class ModelParams(object):
    def __init__(self, args):
        self.d_learning_rate = args.d_learning_rate
        self.g_learning_rate = args.g_learning_rate
        self.batch_size = args.batch_size
        self.d_l2_reg = args.d_l2_reg
        self.input_mean = args.input_mean
        self.input_stddev = args.input_stddev


class GanNormalModel(object):
    def __init__(self, hparams):
        self.generator_input = self.generator_input(hparams)
        with tf.variable_scope("generator"):
            self.generated = self.generator(self.generator_input)

        self.real_input = self.data_batch(hparams)
        with tf.variable_scope("discriminator"):
            self.real_ratings = self.discriminator(self.real_input)

        with tf.variable_scope("discriminator", reuse=True):
            self.generated_ratings = self.discriminator(self.generated)

        self.loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(self.real_ratings), logits=self.real_ratings))
        self.loss_generated = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(self.generated_ratings), logits=self.generated_ratings))
        self.discriminator_loss = self.loss_generated + self.loss_real
        self.generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(self.generated_ratings), logits=self.generated_ratings))



        with tf.variable_scope("generator") as scope:
            generator_variables = [v for v in tf.global_variables() if v.name.startswith(scope.name)]
        with tf.variable_scope("discriminator") as scope:
            discriminator_variables = [v for v in tf.global_variables() if v.name.startswith(scope.name)]


        self.discriminator_loss += hparams.d_l2_reg * add_n([tf.nn.l2_loss(v) for v in discriminator_variables])
        self.discriminator_train = tf.train.AdamOptimizer(hparams.d_learning_rate).minimize(
            self.discriminator_loss, var_list=discriminator_variables, name="train_discriminator")
        self.generator_train = tf.train.AdamOptimizer(hparams.g_learning_rate).minimize(
            self.generator_loss, var_list=generator_variables, name="train_generator")

        self.average_probability_real = tf.reduce_mean(tf.sigmoid(self.real_ratings))
        self.average_probability_fake = tf.reduce_mean(tf.sigmoid(self.generated_ratings))

        tf.summary.scalar("D/cost", self.discriminator_loss)
        tf.summary.scalar("G/cost", self.generator_loss)
        tf.summary.scalar("P(real|1)/", self.average_probability_real)
        tf.summary.scalar("P(real|0)/", self.average_probability_fake)
        self.summaries = tf.summary.merge_all()

    def discriminator(self, input):
        input_size = 1
        features = 256
        weights = tf.get_variable("weights_1", initializer=tf.truncated_normal([input_size, features], stddev=0.1))
        biases = tf.get_variable("biases_1", initializer=tf.constant(0.1, shape=[features]))
        hidden_layer = tf.nn.dropout(tf.nn.relu(tf.matmul(input, weights) + biases), 0.5)

        features = 256
        weights = tf.get_variable("weights_2", initializer=tf.truncated_normal([input_size, features], stddev=0.1))
        biases = tf.get_variable("biases_2", initializer=tf.constant(0.1, shape=[features]))
        hidden_layer = tf.nn.dropout(tf.nn.relu(tf.matmul(input, weights) + biases), 0.5)

        classes = 1
        weights = tf.get_variable("weights_out", initializer=tf.truncated_normal([features, classes], stddev=0.1))
        biases = tf.get_variable("biases_out", initializer=tf.constant(0.1, shape=[classes]))
        return tf.matmul(hidden_layer, weights) + biases

    def generator(self, input):
        self.mean = tf.Variable(tf.constant(0.))
        self.stddev = tf.Variable(tf.constant(1.))
        return input * tf.sqrt(self.stddev ** 2) + self.mean

    def data_batch(self, hparams):
        return tf.random_normal([hparams.batch_size, 1], hparams.input_mean, hparams.input_stddev)

    def generator_input(self, hparams):
        return tf.random_normal([hparams.batch_size, 1], 0., 1.)

def print_graph(session, model, step):
    real, fake, mean, stddev = session.run([
        model.average_probability_real, model.average_probability_fake, model.mean, model.stddev
    ])
    step += 1

    print(
        "Saved model with step %d; real = %f, fake = %f, mean = %f, stddev = %f" % (
            step, real, fake, mean, stddev))


def main(args):
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
    args = parser.parse_args(args)

    model = GanNormalModel(ModelParams(args))

    saver = tf.train.Saver()
    with tf.Session() as session:
        if args.load_checkpoint:
            saver.restore(session, args.load_checkpoint)
        else:
            session.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(args.summaries_dir + '/train', session.graph)

        step = 0
        while True:
            print_graph(session, model, step)
            session.run(model.discriminator_train)
            session.run(model.generator_train)
            summary_writer.add_summary(session.run(model.summaries), step)
            step += 1
            # saver.save(session, "%s/model.ckpt-%d" % (args.train_dir, step))

if __name__ == "__main__":
    main(sys.argv[1:])
