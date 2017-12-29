import numpy as np
import tensorflow as tf


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
