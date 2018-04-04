import tensorflow as tf

from common.math import add_n


class TrainingParams(object):
    """
    All training hyperparameters that should be configured from command line should go here.
    """
    def __init__(self, args):
        self.learning_rate = args.learning_rate
        self.l2_reg = args.l2_reg


class LinearRegressionModel(object):
    """
    Linear Regression model -- a linear model for regression.
    This version of the model operates input data, generated from a uniform distribution with
    normally distributed noise.
    """
    def __init__(self, dataset, training_params):
        self.variables = []
        self.l2_reg_variables = []

        labels, samples = self.input(dataset)
        predicted_labels = self.predicted_labels(samples)
        self.loss = self.get_loss(labels, predicted_labels, training_params)

        # Train the model with Adam.
        with tf.name_scope("train"):
            self.train = tf.train.AdamOptimizer(training_params.learning_rate).minimize(
                self.loss, var_list=self.variables, name="train")

        # Export summaries.
        with tf.name_scope("summaries"):
            tf.summary.scalar("Loss", self.loss)
            tf.summary.scalar("Params/1", self.param1[0])
            tf.summary.scalar("Params/2", self.param2[0])
            self.summaries = tf.summary.merge_all()

    def input(self, dataset):
        with tf.name_scope("input"):
            return dataset.get_next()

    def predicted_labels(self, input):
        """
        The linear regression network.
        """
        with tf.variable_scope("model"):
            output_size = 1
            self.param1 = tf.get_variable("weights", initializer=tf.truncated_normal([output_size], stddev=0.1))
            self.variables.append(self.param1)
            self.l2_reg_variables.append(self.param1)
            self.param2 = tf.get_variable("biases", initializer=tf.constant(0.1, shape=[output_size]))
            self.variables.append(self.param2)
            return input * self.param1 + self.param2

    def get_loss(self, labels, predicted_labels, training_params):
        # Compute the loss funstion -- L2 distance between real and predicted labels, which is equal to the
        # cross-entropy between real and predicted labels.
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.square(labels - predicted_labels))

            # Add optional L2 regularization.
            if training_params.l2_reg != 0.0:
                loss += training_params.l2_reg * add_n([tf.nn.l2_loss(v) for v in self.l2_reg_variables])

            return loss
