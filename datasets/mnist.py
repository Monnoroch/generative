"""
MNIST dataset.
"""

import math
import os
from PIL import Image

import numpy
import tensorflow as tf


def small_mnist_dataset(target_directory, train=True, factor=2):
    """
    Create small MNIST dataset (images 4x smaller than originals).

    Arguments:
    - train - whether to load train or test files.
    """
    images, labels = _load_mnist(target_directory, train)
    new_images = []
    for image in images:
        image_size = int(math.sqrt(image.shape[0]))
        image = Image.frombuffer("F", (image_size, image_size), numpy.getbuffer(image), "raw", "F", 0, 1).resize(
            (image_size/factor, image_size/factor), Image.ANTIALIAS)
        new_images.append(numpy.frombuffer(image.tobytes(), numpy.float32))
    images = numpy.array(new_images)
    return tf.data.Dataset.from_tensor_slices((images, labels))


def mnist_dataset(target_directory, train=True):
    """
    Create MNIST dataset.

    Arguments:
    - train - whether to load train or test files.
    """
    return tf.data.Dataset.from_tensor_slices(_load_mnist(target_directory, train))


def _load_mnist(target_directory, train=True):
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
    return images, labels
