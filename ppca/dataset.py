import tensorflow as tf


class DatasetParams(object):
    """
    All dataset hyperparameters that should be configured from command line should go here.
    """
    def __init__(self, args):
        self.input_mean = args.input_mean
        self.input_stddev = args.input_stddev
        self.batch_size = args.batch_size


def normal_samples(params):
    """
    The input batch generator.
    """
    def example():
        return tf.contrib.distributions.MultivariateNormalDiag(
            params.input_mean, params.input_stddev).sample(sample_shape=[1])[0]
    return tf.data.Dataset.from_tensors([0.]).repeat().map(lambda x: example()).batch(params.batch_size)
