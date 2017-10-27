import tensorflow as tf


def add_n(arr):
  if not arr:
    return 0
  return tf.add_n(arr)


class DatasetParams(object):
    """
    All dataset hyperparameters that should be configured from command line should go here.
    """
    def __init__(self, args):
        self.input_param1 = args.input_param1
        self.input_param2 = args.input_param2


class TrainingParams(object):
    """
    All training hyperparameters that should be configured from command line should go here.
    """
    def __init__(self, args, training):
        self.training = training
        if not training:
            return
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.l2_reg = args.l2_reg


class LinearRegressionModel(object):
    """
    Linear Regression model -- a linear model for regression.
    This version of the model operates input data, generated from a uniform distribution with
    normally distributed noise.
    """
    def __init__(self, dataset_params, training_params):
        # Set up the global step.
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.increment_global_step = tf.assign_add(self.global_step, 1)

        if not training_params.training:
            return

        # Compute the loss funstion -- L2 distance between real and predicted labels, which is equal to the
        # cross-entropy between real and predicted labels.
        labels, samples = self.input_batch(dataset_params, training_params.batch_size)
        with tf.variable_scope("network") as scope:
            predicted_labels = self.predicted_labels(samples)
            variables = [v for v in tf.global_variables() if v.name.startswith(scope.name)]

        self.loss = tf.reduce_mean(tf.square(labels - predicted_labels))

        # Add optional L2 regularization.
        if training_params.l2_reg != 0.0:
            self.loss += training_params.l2_reg * add_n([tf.nn.l2_loss(v) for v in variables])

        # Train the model with Adam.
        self.train = tf.train.AdamOptimizer(training_params.learning_rate).minimize(
            self.loss, var_list=variables, name="train")

        # Export summaries.
        tf.summary.scalar("Loss", self.loss)
        tf.summary.scalar("Params/1", self.param1[0])
        tf.summary.scalar("Params/2", self.param2[0])
        self.summaries = tf.summary.merge_all()

    def predicted_labels(self, input):
        """
        The linear regression network.
        """
        output_size = 1
        self.param1 = tf.get_variable("weights", initializer=tf.truncated_normal([output_size], stddev=0.1))
        self.param2 = tf.get_variable("biases", initializer=tf.constant(0.1, shape=[output_size]))
        return input * self.param1 + self.param2

    def input_batch(self, dataset_params, batch_size):
        """
        The input batch generator.
        """
        samples = tf.random_uniform([batch_size], 0., 10.)
        noise = tf.random_normal([batch_size], mean=0., stddev=1.)
        labels = dataset_params.input_param1 * samples + dataset_params.input_param2 + noise
        return labels, samples
