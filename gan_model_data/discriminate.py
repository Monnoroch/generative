import argparse
import os
import sys

import tensorflow as tf

from gan_model_data import model
from common.hparams import load_hparams


def main(args):
    """
    The main function to sample from the model.
    """
    parser = argparse.ArgumentParser(description="Train the gan-normal model.")
    parser.add_argument("--load_checkpoint", required=True, help="Continue training from a checkpoint")
    parser.add_argument("--samples", type=int, default=1, help="The number of samples to generate")
    args = parser.parse_args(args)

    args.experiment_dir = os.path.dirname(os.path.dirname(os.path.dirname(args.load_checkpoint)))

    # Create the model.
    hparams = model.ModelParams(load_hparams(args))
    model_ops = model.GanNormalModel(None, None, model.TrainingParams(None, training=False))

    input = tf.placeholder(dtype=tf.float32, shape=(1, 1))
    discriminated = model_ops.discriminate(input, hparams)

    saver = tf.train.Saver()
    with tf.Session() as session:
        # Initializing the model using a saved checkpoint.
        saver.restore(session, args.load_checkpoint)

        global_step = session.run(model_ops.global_step)
        print("Load model at step %d." % global_step)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        started = False

        for line in sys.stdin:
            if not started:
                if line.startswith("~~~~~~~"):
                    started = True
                continue
            example = float(line)
            result = session.run(discriminated, feed_dict={input: [[example]]})
            print("%f\t%s" % (result[0][0], result[0][0] > 0))


if __name__ == "__main__":
    main(sys.argv[1:])
