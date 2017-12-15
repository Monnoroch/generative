import tensorflow as tf


class DatasetParams(object):
    """
    All dataset hyperparameters that should be configured from command line should go here.
    """
    def __init__(self, args):
        self.input_mean = args.input_mean
        self.input_stddev = args.input_stddev
        self.batch_size = args.batch_size


def gmm_classes(params):
    """
    The input batch generator.
    """
    def example():
        input_mean = tf.constant(params.input_mean, dtype=tf.float32)
        input_stddev = tf.constant(params.input_stddev, dtype=tf.float32)
        count = len(params.input_mean)
        labels = tf.contrib.distributions.Categorical(
            probs=[1./count] * count).sample(sample_shape=[params.batch_size])
        components = []
        for i in range(params.batch_size):
            components.append(
                tf.contrib.distributions.Normal(
                    loc=input_mean[labels[i]],
                    scale=input_stddev[labels[i]]).sample(sample_shape=[1]))
        samples = tf.concat(components, 0)
        return labels, samples
    return tf.data.Dataset.from_tensors([0.]).repeat().map(lambda x: example())
