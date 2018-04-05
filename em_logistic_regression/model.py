import tensorflow as tf

from common.math import add_n

from logistic_regression.model import LogisticRegressionModel


class UnsupervisedLogisticRegressionModel(LogisticRegressionModel):
    """
    EM-based logistic Regression model -- a linear model for binary classification.
    """
    def __init__(self, dataset, training_params):
        self.variables = []
        self.l2_reg_variables = []

        labels, samples = self.input(dataset)
        predicted_logits = self.discriminator(samples)
        latent_variables = self.get_latent(predicted_logits, training_params)
        self.expected_loss = self.get_loss(latent_variables, predicted_logits, training_params)
        self.real_loss = self.get_loss(labels, predicted_logits, training_params)
        self.loss = self.expected_loss

        accuracy = self.accuracy(labels, predicted_logits, samples)

        # Train the model with Adam.
        with tf.name_scope("train"):
            self.train = tf.train.AdamOptimizer(training_params.learning_rate).minimize(
                self.loss, var_list=self.variables, name="train")

        # Export summaries.
        with tf.name_scope("summaries"):
            tf.summary.scalar("Loss on expected labels", self.expected_loss)
            tf.summary.scalar("Loss", self.real_loss)
            tf.summary.scalar("Accuracy", accuracy)
            self.summaries = tf.summary.merge_all()

    def get_latent(self, predicted_logits, training_params):
        """
        Transform visible variables into latent space.
        """
        class_1_probs = tf.expand_dims(tf.nn.sigmoid(predicted_logits), 1)
        class_0_probs = 1. - class_1_probs
        class_probs = tf.concat([class_0_probs, class_1_probs], 1)
        latent = []
        for value in tf.unstack(class_probs, training_params.batch_size):
            latent.append(tf.contrib.distributions.Categorical(probs=value).sample())
        latent = tf.stack(latent)
        return tf.stop_gradient(latent)
