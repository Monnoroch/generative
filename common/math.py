import tensorflow as tf


def add_n(arr):
  if not arr:
    return 0
  return tf.add_n(arr)


def product(values):
    result = 1
    for value in values:
        result *= value.value
    return result
