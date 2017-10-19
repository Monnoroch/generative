import tensorflow as tf


def add_n(arr):
  if not arr:
    return 0
  return tf.add_n(arr)


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
        self.learning_rate = args.learning_rate
        self.l2_reg = args.l2_reg


class LogisticRegressionModel(object):
    def __init__(self, dataset_params, training_params):
        # Set up the global step.
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.increment_global_step = tf.assign_add(self.global_step, 1)

        if not training_params.training:
            return

        labels, samples = self.input_batch(dataset_params, training_params.batch_size)
        with tf.variable_scope("discriminator") as scope:
            predicted_labels = self.discriminator(samples)
            variables = [v for v in tf.global_variables() if v.name.startswith(scope.name)]

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(labels, tf.float32), logits=predicted_labels))
        if training_params.l2_reg != 0.0:
            self.loss += training_params.l2_reg * add_n([tf.nn.l2_loss(v) for v in variables])

        self.train = tf.train.AdamOptimizer(
            training_params.learning_rate + tf.cast(self.global_step, dtype=tf.float32) * 1./2000).minimize(
                self.loss, var_list=variables, name="train")

        tf.summary.scalar("Loss", self.loss)
        self.summaries = tf.summary.merge_all()

    def discriminator(self, input):
        output_size = 1
        self.param1 = tf.get_variable("weights", initializer=tf.truncated_normal([output_size], stddev=0.1))
        self.param2 = tf.get_variable("biases", initializer=tf.constant(0.1, shape=[output_size]))
        return input * self.param1 + self.param2

    def input_batch(self, dataset_params, batch_size):
        input_mean = tf.constant(dataset_params.input_mean, dtype=tf.float32)
        input_stddev = tf.constant(dataset_params.input_stddev, dtype=tf.float32)
        count = len(dataset_params.input_mean)
        labels = tf.contrib.distributions.Categorical(
            probs=[1./count] * count).sample(sample_shape=[batch_size])
        components = []
        for i in range(batch_size):
            components.append(
                tf.contrib.distributions.Normal(
                    loc=input_mean[labels[i]],
                    scale=input_stddev[labels[i]]).sample(sample_shape=[1]))
        samples = tf.concat(components, 0)
        return labels, samples
