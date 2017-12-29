import tensorflow as tf

from common.math import add_n, product

class ModelParams(object):
    """
    All model hyperparameters that should be configured from command line should go here.
    """
    def __init__(self, args):
        self.dropout = args.dropout
        self.generator_features = args.generator_features
        self.discriminator_features = args.discriminator_features
        self.latent_space_size = args.latent_space_size


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
        self.optimizer = args.optimizer
        self.smooth_labels = args.smooth_labels


class GanModel(object):
    """
    A GAN that fits a one-dimentional input data that has a normal distribution.

    The input space X is just R.
    The input data probability distribution is N(mean, stddev).
    The generator knows that the data is normally distributed and tries to find the parameters of the normal
        distribution, so it's not a neural network but rather just a linear mapping. The generator input is a random
        variable distributed as N(0, 1).
    The discriminator however is a deep fully connected NN. Dropout and weight decay are uesd to reguralize the model.
    """
    def __init__(self, model_params, dataset, training_params):
        # Set up the global step.
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.increment_global_step = tf.assign_add(self.global_step, 1)

        if not training_params.training:
            return

        # Get the real and the fake inputs.
        self.real_input = dataset.get_next()
        input_shape = self.real_input[0].shape
        self.generator_input_batch = self.generator_input(training_params.batch_size, [model_params.latent_space_size])
        with tf.variable_scope("generator"):
            self.generated = self.generator(
                self.generator_input_batch, input_shape, model_params)

        with tf.variable_scope("generator", reuse=True):
            sample_side_size = 8
            samples = sample_side_size**2
            batches = int(samples / training_params.batch_size) + 1
            real_input_sample = tf.concat(list(map(lambda _: dataset.get_next(), range(batches))), axis=0)[:samples,:,:,:]
            generator_input_batch = self.generator_input(samples, [model_params.latent_space_size])
            generated_sample = self.generator(generator_input_batch, input_shape, model_params)

        # Get discriminator logits for both inputs.
        with tf.variable_scope("discriminator"):
            self.real_ratings = self.discriminator(self.real_input, model_params)
        # We want to share parameters between discriminator instances because logically there's only one discriminator.
        with tf.variable_scope("discriminator", reuse=True):
            self.generated_ratings = self.discriminator(self.generated, model_params)

        # Discriminator loss minimizes the discrimination error on both real and fake inputs.
        labels = tf.ones_like(self.real_ratings)
        if training_params.smooth_labels:
            labels = labels * 0.9
        self.loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=self.real_ratings))
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

        tf.summary.scalar("Cost/D", self.discriminator_loss)
        tf.summary.scalar("Cost/G", self.generator_loss)
        tf.summary.scalar("P/real_on_real", self.average_probability_real)
        tf.summary.scalar("P/real_on_fake", self.average_probability_fake)
        tf.summary.image("Real data", image_grid(clip_images(real_input_sample), sample_side_size), max_outputs=1)
        tf.summary.image("Fake data", image_grid(clip_images(generated_sample), sample_side_size), max_outputs=1)
        self.summaries = tf.summary.merge_all()

    def discriminator(self, input, hparams):
        """
        Discriminator is a theee-layer fully connected NN.
        """
        # First fully connected layer with N features and optional dropout.
        batch_size = tf.shape(input)[0]
        input_size = product(input.shape[1:])
        input = tf.reshape(input, [batch_size, input_size])

        new_features = input_size
        hidden_layer = input

        for i in range(len(hparams.discriminator_features)):
            features = new_features
            new_features = hparams.discriminator_features[i]
            weights = tf.get_variable("weights_%d" % i, initializer=tf.truncated_normal([features, new_features], stddev=0.02))
            biases = tf.get_variable("biases_%d" % i, initializer=tf.constant(0., shape=[new_features]))
            hidden_layer = tf.nn.relu(tf.matmul(hidden_layer, weights) + biases)
            if hparams.dropout != 0.0:
                hidden_layer = tf.nn.dropout(hidden_layer, hparams.dropout)

        # Final linear layer to compute the classifier's logits.
        features = new_features
        output_size = 1
        weights = tf.get_variable("weights_out", initializer=tf.truncated_normal([features, output_size], stddev=0.02))
        biases = tf.get_variable("biases_out", initializer=tf.constant(0., shape=[output_size]))
        return tf.matmul(hidden_layer, weights) + biases

    def generator(self, input, output_shape, hparams):
        """
        Generator is a deep neural network with fully connected layers.
        """
        # First fully connected layer with N features.
        batch_size = input.shape[0].value
        input_size = input.shape[1].value

        new_features = input_size
        hidden_layer = input

        for i in range(len(hparams.generator_features)):
            features = new_features
            new_features = hparams.generator_features[i]
            weights = tf.get_variable("weights_%d" % i, initializer=tf.truncated_normal([features, new_features], stddev=0.02))
            biases = tf.get_variable("biases_%d" % i, initializer=tf.constant(0., shape=[new_features]))
            hidden_layer = tf.nn.relu(tf.matmul(hidden_layer, weights) + biases)

        # Final linear layer to generate the example.
        features = new_features
        output_size = product(output_shape)
        weights = tf.get_variable("weights_out", initializer=tf.truncated_normal([features, output_size], stddev=0.02))
        biases = tf.get_variable("biases_out", initializer=tf.constant(0., shape=[output_size]))
        return tf.reshape(tf.matmul(hidden_layer, weights) + biases, tf.concat([[batch_size], output_shape], axis=0))

    def generator_input(self, samples, output_shape):
        """
        Generator input data are just samples from N(0, 1).
        """
        return tf.random_normal(tf.concat([[samples], output_shape], 0), 0., 1.)


def clip_images(images):
    return tf.minimum(tf.maximum(images, 0.0), 1.0)


def image_grid(image_batch, max_side_size, batch_size=None):
    """Make a grid from a batch of images.

    Concatenate first max_side_size*max_side_size images from a batch in a
    grid of size max_side_size x max_side_size.

    Args:
        batch: a tensor batch of images.
        batch_size: a int size of this batch.
        max_side_size: the size of the grid.
    Returns:
        A grid tensor.
    """
    if batch_size is None:
        side_size = max_side_size
    else:
        side_size = min(int(np.floor(np.log2(batch_size))), max_side_size)
    start = 0
    rows = []
    while start < side_size**2:
        t = tf.concat([image_batch[i, :, :, :] for i in range(start, start + side_size)], 1)
        rows.append(t)
        start += side_size
    return tf.expand_dims(tf.concat(rows, 0), 0)
