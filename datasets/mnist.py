"""
MNIST dataset.
"""

import os

import tensorflow as tf


def mnist_dataset(target_directory, train=True):
    """
    Create MNIST dataset.

    Arguments:
    - train - whether to load train or test files.
    """
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    from tensorflow.examples.tutorials.mnist import input_data
    dataset = input_data.read_data_sets(target_directory, one_hot=True, validation_size=0)
    if train:
        images = dataset.train.images
        labels = dataset.train.labels
    else:
        images = dataset.test.images
        labels = dataset.test.labels
    return tf.contrib.data.Dataset.from_tensor_slices((images, labels))
