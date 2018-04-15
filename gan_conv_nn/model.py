import tensorflow as tf

from common.math import add_n, prod, product
from common.summary import clip_images, image_grid
from gan_deep_nn.model import GanModel


class ModelParams(object):
    """
    All model hyperparameters that should be configured from command line should go here.
    """
    def __init__(self, args):
        self.dropout = args.dropout
        self.generator_features = args.generator_features
        self.discriminator_features = args.discriminator_features
        self.latent_space_size = args.latent_space_size
        self.discriminator_filter_sizes = args.discriminator_filter_sizes
        self.generator_filter_sizes = args.generator_filter_sizes
        self.leaky_relu = args.leaky_relu
        self.discriminator_strides = args.discriminator_strides
        self.generator_strides = args.generator_strides


def relu(x, hparams):
    if hparams.leaky_relu != 0.0:
        f1 = 0.5 * (1 + hparams.leaky_relu)
        f2 = 0.5 * (1 - hparams.leaky_relu)
        return f1 * x + f2 * tf.abs(x)
    return tf.nn.relu(x)


def get_initial_stride(stride, layers, output_size):
    combined_stride = stride**(layers - 1)
    if output_size % combined_stride:
        raise Error("Generator output image size has to be divisible by the stride**(number of layers - 1)")
    return output_size / combined_stride


class ConvGanModel(GanModel):
    def discriminator(self, input, hparams):
        """
        Discriminator is a fully-convolutional NN.
        """
        input_size = 1

        new_features = input_size
        hidden_layer = input

        for i in range(len(hparams.discriminator_features)):
            features = new_features
            new_features = hparams.discriminator_features[i]
            filter_size = hparams.discriminator_filter_sizes[i]
            filters = tf.get_variable("weights_%d" % i,
                initializer=tf.constant(0., shape=[filter_size, filter_size, features, new_features]))
            biases = tf.get_variable("biases_%d" % i, initializer=tf.constant(0., shape=[new_features]))
            stride = hparams.discriminator_strides[i]
            hidden_layer = tf.nn.conv2d(hidden_layer, filters, strides=[1, stride, stride, 1], padding="SAME")
            hidden_layer = relu(hidden_layer + biases, hparams)
            if hparams.dropout != 0.0:
                hidden_layer = tf.nn.dropout(hidden_layer, hparams.dropout, shape=[input.shape[0], 1, 1, input.shape[3]])

        # Final linear layer to compute the classifier's logits.
        features = product(hidden_layer.shape[1:])
        hidden_layer = tf.reshape(hidden_layer, [None, features])
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

        hidden_layer = tf.expand_dims(tf.expand_dims(input, 1), 1)
        new_features = input_size

        if prod(hparams.generator_strides) != output_shape[1]:
            raise Error("The total generator stride must be equal to the output shape")

        for i in range(len(hparams.generator_features)):
            features = new_features
            new_features = hparams.generator_features[i]
            biases = tf.get_variable("biases_%d" % i, initializer=tf.constant(0., shape=[new_features]))
            filter_size = hparams.generator_filter_sizes[i]
            filters = tf.get_variable("weights_%d" % i,
                initializer=tf.constant(0., shape=[filter_size, filter_size, new_features, features]))
            stride = hparams.generator_strides[i]
            new_shape = [hidden_layer.shape[0], hidden_layer.shape[1] * stride, hidden_layer.shape[2] * stride, new_features]
            hidden_layer = tf.nn.conv2d_transpose(
                hidden_layer, filters, output_shape=new_shape, strides=[1, stride, stride, 1], padding="SAME")
            hidden_layer = relu(hidden_layer + biases, hparams)

        return hidden_layer
