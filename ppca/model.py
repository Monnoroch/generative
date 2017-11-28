import math
import tensorflow as tf


def add_n(arr):
  if not arr:
    return 0
  return tf.add_n(arr)


class TrainingParams(object):
    """
    All training hyperparameters that should be configured from command line should go here.
    """
    def __init__(self, args):
        self.learning_rate = args.learning_rate
        self.l2_reg = args.l2_reg


class ModelParams(object):
    """
    All model hyperparameters that should be configured from command line should go here.
    """
    def __init__(self, args):
        self.latent_space_size = args.latent_space_size


class PpcaModel(object):
    """
    Probabilistic Principal Compoent Analysis model -- a linear latent variable model for generating samples.
    """
    def __init__(self, dataset, hparams, training_params, batch_size):
        # Set up the global step.
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.increment_global_step = tf.assign_add(self.global_step, 1)

        # Compute the loss funstion -- joint maximum likelihood of visible and latent vasiables.
        sample = dataset.get_next()
        input_size = sample.shape[1].value

        self.stddev = tf.get_variable("stddev", initializer=tf.constant(0.1, shape=[1]))
        self.stddev_squared = self.stddev**2
        self.biases = tf.get_variable("biases", initializer=tf.constant(0.1, shape=[input_size]))
        self.weights = tf.get_variable(
            "weights", initializer=tf.truncated_normal([input_size, hparams.latent_space_size], stddev=0.1))

        latent_sample = self.get_latent(sample, hparams, batch_size)
        norm_squared = tf.reduce_sum(
            (sample - self.biases - batch_matmul(self.weights, latent_sample, batch_size))**2, axis=1)
        self.loss = (tf.reduce_mean(
            input_size * tf.log(self.stddev_squared) + 1./self.stddev_squared * norm_squared))

        variables = [self.biases, self.weights, self.stddev]
        l2_reg_variables = [self.weights]

        # Add optional L2 regularization.
        if training_params.l2_reg != 0.0:
            self.loss += training_params.l2_reg * add_n([tf.nn.l2_loss(v) for v in l2_reg_variables])

        # Train the model with Adam.
        self.train = tf.train.AdamOptimizer(training_params.learning_rate).minimize(
            self.loss, var_list=variables, name="train")

        self.data_dist_mean = self.biases
        self.data_dist_stddev = tf.matmul(
            self.weights, self.weights, transpose_b=True) + self.stddev_squared * tf.eye(input_size)

        learned_sample = tf.contrib.distributions.MultivariateNormalFullCovariance(
            self.data_dist_mean, self.data_dist_stddev).sample(sample_shape=[1024])

        # Export summaries.
        tf.summary.scalar("Loss/Log", self.loss)
        tf.summary.scalar("Stddev", self.stddev[0])
        for i in range(input_size):
            tf.summary.scalar("Bias/%d" % i, self.biases[i])
        for i in range(input_size):
            tf.summary.histogram("Sample/Real_%d" % i, sample[:, i])
        for i in range(input_size):
            tf.summary.histogram("Sample/Learned_%d" % i, learned_sample[:, i])
        tf.summary.histogram("LatentSample/Learned", latent_sample)
        tf.summary.histogram("LatentSample/Real", tf.random_normal([1024, 1], 0., 1.))
        for i in range(input_size):
            for j in range(hparams.latent_space_size):
                tf.summary.scalar("Weights/%d_%d" % (i, j), self.weights[i][j])
        self.summaries = tf.summary.merge_all()

    def get_latent(self, input, hparams, batch_size):
        """
        Transform visible variables into latent space.
        """
        identity = tf.eye(hparams.latent_space_size)
        matrix = tf.matrix_inverse(
            tf.matmul(self.weights, self.weights, transpose_a=True) + self.stddev_squared * identity)
        mean_matrix = tf.matmul(matrix, self.weights, transpose_b=True)
        expected_latent = batch_matmul(mean_matrix, input - self.biases, batch_size)
        stddev_matrix = self.stddev_squared * matrix
        noise = tf.contrib.distributions.MultivariateNormalFullCovariance(
            tf.zeros(hparams.latent_space_size), stddev_matrix).sample(sample_shape=[batch_size])
        return tf.stop_gradient(expected_latent + noise)


def batch_matmul(matrix, vector_batch, batch_size):
    output_size = matrix.shape[0].value
    vector_batch = tf.expand_dims(vector_batch, 2)
    return tf.reshape(
        tf.stack(map(lambda vector: tf.matmul(matrix, vector), tf.unstack(vector_batch, num=batch_size))),
        [-1, output_size])
