import argparse
import sys

import tensorflow as tf

from gan_model_data import model
from common.experiment import Experiment


def main(args):
    """
    The main function to sample from the model.
    """
    parser = argparse.ArgumentParser(description="Train the gan-normal model.")
    parser.add_argument("--load_checkpoint", required=True, help="Continue training from a checkpoint")
    args = parser.parse_args(args)

    experiment = Experiment.from_checkpoint(args.load_checkpoint)
    hparams = experiment.load_hparams(model.ModelParams)

    # Create the model.
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
            print("%f\t%s" % (result[0][0], result[0][0] > 0.5))


if __name__ == "__main__":
    main(sys.argv[1:])
