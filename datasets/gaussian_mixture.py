import tensorflow as tf


class DatasetParams(object):
    """
    All dataset hyperparameters that should be configured from command line should go here.
    """
    def __init__(self, args):
        if len(args.input_mean) != len(args.input_stddev):
            raise "There must be the same number of input means and standard deviations."

        self.input_mean = args.input_mean
        self.input_stddev = args.input_stddev

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--input_mean", type=float, default=[15.], action="append",
            help="The mean of the dataset mode. The dataset can have multiple modes")
        parser.add_argument("--input_stddev", type=float, default=[7.], action="append",
            help="The standard deviation of the dataset mode. The dataset can have multiple modes")

def dataset(params):
    """
    The input batch generator.
    """
    with tf.name_scope("dataset"):
        def sample():
            input_mean = tf.constant(params.input_mean, dtype=tf.float32)
            input_stddev = tf.constant(params.input_stddev, dtype=tf.float32)
            count = len(params.input_mean)
            label = tf.contrib.distributions.Categorical(probs=[1./count] * count).sample()
            example = tf.contrib.distributions.Normal(
                loc=input_mean[label],
                scale=input_stddev[label]).sample()
            return label, example
        return tf.data.Dataset.from_tensors([0.]).repeat().map(lambda x: sample())
