import tensorflow as tf

from common.math import add_n


class ModelParams(object):
    """
    All model hyperparameters that should be configured from command line should go here.
    """
    def __init__(self, args):
        self.dropout = args.dropout
        self.nn_generator = args.nn_generator
        self.generator_features = args.generator_features
        self.discriminator_features = args.discriminator_features


class DatasetParams(object):
    """
    All dataset hyperparameters that should be configured from command line should go here.
    """
    def __init__(self, args):
        self.input_mean = args.input_mean
        self.input_stddev = args.input_stddev


class TrainingParams(object):
    """
    All training hyperparameters that should be configured from command line should go here.
    """
    def __init__(self, args, training):
        self.training = training
        if not training:
            return
        self.batch_size = args.batch_size
        self.d_learning_rate = args.d_learning_rate
        self.g_learning_rate = args.g_learning_rate
        self.d_l2_reg = args.d_l2_reg
        self.g_l2_reg = args.g_l2_reg


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
    def __init__(self, model_params, dataset_params, training_params):
        if not training_params.training:
            return

        # Get the real and the fake inputs.
        self.generator_input = self.generator_input(training_params.batch_size)
        self.real_input = self.data_batch(dataset_params, training_params.batch_size)
        self.generated = self.generator(self.generator_input, model_params)

        # Get discriminator logits for both inputs.
        with tf.variable_scope("discriminator"):
            self.real_ratings = self.discriminator(self.real_input, model_params)
        # We want to share parameters between discriminator instances because logically there's only one discriminator.
        with tf.variable_scope("discriminator", reuse=True):
            self.generated_ratings = self.discriminator(self.generated, model_params)

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
            l2_reg_generator_variables = [v for v in generator_variables if v.name.find("weights") != -1]
        with tf.variable_scope("discriminator") as scope:
            discriminator_variables = [v for v in tf.global_variables() if v.name.startswith(scope.name)]
            l2_reg_discriminator_variables = [v for v in discriminator_variables if v.name.find("weights") != -1]

        # Optionally add L2 regularization to the discriminator.
        if training_params.d_l2_reg != 0.0:
            self.discriminator_loss += training_params.d_l2_reg * add_n(
                [tf.nn.l2_loss(v) for v in l2_reg_discriminator_variables])
        if training_params.g_l2_reg != 0.0:
            self.generator_loss += training_params.g_l2_reg * add_n(
                [tf.nn.l2_loss(v) for v in l2_reg_generator_variables])

        # Optimize losses with Adam optimizer.
        self.generator_train = tf.train.AdamOptimizer(training_params.g_learning_rate).minimize(
            self.generator_loss, var_list=generator_variables, name="train_generator")
        self.discriminator_train = tf.train.AdamOptimizer(training_params.d_learning_rate).minimize(
            self.discriminator_loss, var_list=discriminator_variables, name="train_discriminator")

        # Add useful graphs to Tensorboard.
        self.average_probability_real = tf.reduce_mean(tf.sigmoid(self.real_ratings))
        self.average_probability_fake = tf.reduce_mean(tf.sigmoid(self.generated_ratings))

        tf.summary.scalar("D/cost", self.discriminator_loss)
        tf.summary.scalar("G/cost", self.generator_loss)
        tf.summary.scalar("P_real_on_real/", self.average_probability_real)
        tf.summary.scalar("P_real_on_fake/", self.average_probability_fake)
        if not model_params.nn_generator:
            tf.summary.scalar("G/mean", self.mean)
            tf.summary.scalar("G/stddev", self.stddev)
        tf.summary.histogram("Real data", self.real_input)
        tf.summary.histogram("Fake data", self.generated)
        self.summaries = tf.summary.merge_all()

    def discriminator(self, input, hparams):
        """
        Discriminator is a theee-layer fully connected NN.
        """
        # First fully connected layer with N features and optional dropout.

        input_size = 1
        new_features = input_size
        hidden_layer = input

        for i in range(len(hparams.discriminator_features)):
            features = new_features
            new_features = hparams.discriminator_features[i]
            weights = tf.get_variable("weights_%d" % i, initializer=tf.truncated_normal([features, new_features], stddev=0.1))
            biases = tf.get_variable("biases_%d" % i, initializer=tf.constant(0.1, shape=[new_features]))
            hidden_layer = tf.nn.relu(tf.matmul(hidden_layer, weights) + biases)
            if hparams.dropout != 0.0:
                hidden_layer = tf.nn.dropout(hidden_layer, hparams.dropout)

        # Final linear layer to compute the classifier's logits.
        output_size = 1
        features = new_features
        weights = tf.get_variable("weights_out", initializer=tf.truncated_normal([features, output_size], stddev=0.1))
        biases = tf.get_variable("biases_out", initializer=tf.constant(0.1, shape=[output_size]))
        return tf.matmul(hidden_layer, weights) + biases

    def discriminate(self, input, hparams):
        with tf.variable_scope("discriminator"):
            return tf.nn.sigmoid(self.discriminator(input, hparams))

    def generator(self, input, hparams):
        """
        Generator is just a linear transformation of the input.
        """
        with tf.variable_scope("generator"):
            if not hparams.nn_generator:
                self.mean = tf.Variable(tf.constant(0.), name="mean")
                # Standard deviation has to be positive, so we make sure it is by computing the absolute value of the variable.
                self.stddev = tf.sqrt(tf.Variable(tf.constant(1.), name="stddev") ** 2)
                return input * self.stddev + self.mean

            input_size = 1
            new_features = input_size
            hidden_layer = input

            for i in range(len(hparams.generator_features)):
                features = new_features
                new_features = hparams.generator_features[i]
                weights = tf.get_variable("weights_%d" % i, initializer=tf.truncated_normal([features, new_features], stddev=0.1))
                biases = tf.get_variable("biases_%d" % i, initializer=tf.constant(0.1, shape=[new_features]))
                hidden_layer = tf.nn.relu(tf.matmul(hidden_layer, weights) + biases)

            # Final linear layer to generate the example.
            output_size = 1
            features = new_features
            weights = tf.get_variable("weights_out", initializer=tf.truncated_normal([features, output_size], stddev=0.1))
            biases = tf.get_variable("biases_out", initializer=tf.constant(0.1, shape=[output_size]))
            return tf.matmul(hidden_layer, weights) + biases

    def data_batch(self, dataset_params, samples):
        """
        Input data are just samples from N(mean, stddev).
        """
        count = len(dataset_params.input_mean)
        componens = []
        for i in range(count):
            componens.append(
                tf.contrib.distributions.Normal(loc=dataset_params.input_mean[i], scale=dataset_params.input_stddev[i]))
        return tf.contrib.distributions.Mixture(
          cat=tf.contrib.distributions.Categorical(probs=[1./count] * count),
          components=componens).sample(sample_shape=[samples, 1])

    def generator_input(self, samples):
        """
        Generator input data are just samples from N(0, 1).
        """
        return tf.random_normal([samples, 1], 0., 1.)
