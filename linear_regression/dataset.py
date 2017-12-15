import tensorflow as tf


class DatasetParams(object):
    """
    All dataset hyperparameters that should be configured from command line should go here.
    """
    def __init__(self, args):
        self.input_param1 = args.input_param1
        self.input_param2 = args.input_param2
        self.batch_size = args.batch_size


def linear_dependent_with_error(params):
    """
    The input batch generator.
    """
    def example():
        samples = tf.random_uniform([params.batch_size], 0., 10.)
        noise = tf.random_normal([params.batch_size], mean=0., stddev=1.)
        labels = params.input_param1 * samples + params.input_param2 + noise
        return labels, samples
    return tf.data.Dataset.from_tensors([0.]).repeat().map(lambda x: example())
