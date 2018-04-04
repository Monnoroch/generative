import tensorflow as tf

from common.math import add_n


class TrainingParams(object):
    """
    All training hyperparameters that should be configured from command line should go here.
    """
    def __init__(self, args):
        self.learning_rate = args.learning_rate
        self.l2_reg = args.l2_reg
        self.batch_size = args.batch_size

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--batch_size", type=int, default=32,
            help="The size of the minibatch")
        parser.add_argument("--learning_rate", type=float, default=0.01,
            help="The learning rate")
        parser.add_argument("--l2_reg", type=float, default=0.0005,
            help="The L2 regularization parameter")


class LogisticRegressionModel(object):
    """
    Logistic Regression model -- a linear model for binary classification.
    This version of the model operates input data, generated from two normally distributed classes.
    """
    def __init__(self, dataset, training_params):
        self.variables = []
        self.l2_reg_variables = []

        labels, samples = self.input(dataset)
        predicted_labels = self.discriminator(samples)
        self.loss = self.get_loss(labels, predicted_labels, training_params)

        # Train the model with Adam.
        with tf.name_scope("train"):
            self.train = tf.train.AdamOptimizer(training_params.learning_rate).minimize(
                self.loss, var_list=self.variables, name="train")

        # Export summaries.
        with tf.name_scope("summaries"):
            tf.summary.scalar("Loss", self.loss)
            self.summaries = tf.summary.merge_all()

    def input(self, dataset):
        with tf.name_scope("input"):
            return dataset.get_next()

    def discriminator(self, input):
        """
        The discriminator network to classify inputs. Returns logits.
        """
        with tf.variable_scope("model"):
            output_size = 1
            weights = tf.get_variable("weights", initializer=tf.truncated_normal([output_size], stddev=0.1))
            self.variables.append(weights)
            self.l2_reg_variables.append(weights)
            biases = tf.get_variable("biases", initializer=tf.constant(0.1, shape=[output_size]))
            self.variables.append(biases)
            return input * weights + biases

    def get_loss(self, labels, predicted_labels, training_params):
        # Compute the loss funstion -- cross-entropy between real and predicted labels.
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(labels, tf.float32), logits=predicted_labels))

            # Add optional L2 regularization.
            if training_params.l2_reg != 0.0:
                loss += training_params.l2_reg * add_n([tf.nn.l2_loss(v) for v in self.l2_reg_variables])

            return loss
