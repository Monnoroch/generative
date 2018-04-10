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
        latent_variables = self.get_latent(predicted_logits)
        self.loss = self.get_loss(latent_variables, predicted_logits, training_params)

        loss_on_real = self.get_loss(labels, predicted_logits, training_params)
        accuracy = self.accuracy(labels, predicted_logits, samples)
        latents_accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, latent_variables), tf.float32))

        # Train the model with Adam.
        with tf.name_scope("train"):
            self.train = tf.train.AdamOptimizer(training_params.learning_rate).minimize(
                self.loss, var_list=self.variables, name="train")

        # Export summaries.
        with tf.name_scope("summaries"):
            tf.summary.scalar("Loss", loss_on_real)
            tf.summary.scalar("Unsupervised loss", self.loss)
            tf.summary.scalar("Accuracy", accuracy)
            tf.summary.scalar("Latents accuracy vs real labels", latents_accuracy)
            tf.summary.scalar("Weight", self.variables[0][0])
            tf.summary.scalar("Bias", self.variables[1][0])
            self.summaries = tf.summary.merge_all()

    def get_latent(self, predicted_logits):
        """
        Transform visible variables into latent space.
        """
        # latent = []
        # for value in tf.unstack(predicted_logits, training_params.batch_size):
        #     prob_1 = tf.nn.sigmoid(value)
        #     latent.append(tf.contrib.distributions.Categorical(probs=[1. - prob_1, prob_1]).sample())
        # latent = tf.stack(latent)
        # latent = tf.cast(tf.nn.sigmoid(predicted_logits) > 0.5, tf.int32)
        latent = tf.nn.sigmoid(predicted_logits)
        return tf.stop_gradient(latent)
