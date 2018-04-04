import tensorflow as tf

from common.math import add_n

from logistic_regression.model import LogisticRegressionModel

class TrainingParams(object):
    """
    All training hyperparameters that should be configured from command line should go here.
    """
    def __init__(self, args):
        self.learning_rate = args.learning_rate
        self.l2_reg = args.l2_reg


class EmLogisticRegressionModel(LogisticRegressionModel):
    """
    EM-based logistic Regression model -- a linear model for binary classification.
    """
    def __init__(self, dataset, training_params, batch_size):
        # Set up the global step.
        with tf.name_scope("global_step_tools"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.increment_global_step = tf.assign_add(self.global_step, 1)

        self.variables = []
        self.l2_reg_variables = []

        _, samples = self.input(dataset)
        predicted_labels = self.discriminator(samples)
        latent_variables = self.get_latent(predicted_labels, batch_size)
        self.loss = self.get_loss(latent_variables, predicted_labels, training_params)

        # Train the model with Adam.
        with tf.name_scope("train"):
            self.train = tf.train.AdamOptimizer(training_params.learning_rate).minimize(
                self.loss, var_list=self.variables, name="train")

        # Export summaries.
        with tf.name_scope("summaries"):
            tf.summary.scalar("Loss", self.loss)
            self.summaries = tf.summary.merge_all()

    def get_latent(self, class_probs, batch_size):
        """
        Transform visible variables into latent space.
        """
        latent = tf.contrib.distributions.Categorical(probs=class_probs).sample(shape=batch_size)
        return tf.stop_gradient(latent)
