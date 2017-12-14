import tensorflow as tf

from common.math import add_n


class TrainingParams(object):
    """
    All training hyperparameters that should be configured from command line should go here.
    """
    def __init__(self, args, training):
        self.learning_rate = args.learning_rate
        self.l2_reg = args.l2_reg


class LogisticRegressionModel(object):
    """
    Logistic Regression model -- a linear model for binary classification.
    This version of the model operates input data, generated from two normally distributed classes.
    """
    def __init__(self, dataset, training_params):
        # Set up the global step.
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.increment_global_step = tf.assign_add(self.global_step, 1)

        # Compute the loss funstion -- cross-entropy between real and predicted labels.
        labels, samples = dataset.get_next()
        with tf.variable_scope("discriminator") as scope:
            predicted_labels = self.discriminator(samples)
            variables = [v for v in tf.global_variables() if v.name.startswith(scope.name)]
            l2_reg_variables = [v for v in variables if v.name.find("weights") != -1]

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(labels, tf.float32), logits=predicted_labels))

        # Add optional L2 regularization.
        if training_params.l2_reg != 0.0:
            self.loss += training_params.l2_reg * add_n([tf.nn.l2_loss(v) for v in l2_reg_variables])

        # Train the model with Adam.
        self.train = tf.train.AdamOptimizer(training_params.learning_rate).minimize(
            self.loss, var_list=variables, name="train")

        # Export summaries.
        tf.summary.scalar("Loss", self.loss)
        self.summaries = tf.summary.merge_all()

    def discriminator(self, input):
        """
        The discriminator network to classify inputs. Returns logits.
        """
        output_size = 1
        self.param1 = tf.get_variable("weights", initializer=tf.truncated_normal([output_size], stddev=0.1))
        self.param2 = tf.get_variable("biases", initializer=tf.constant(0.1, shape=[output_size]))
        return input * self.param1 + self.param2
