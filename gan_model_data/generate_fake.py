import argparse
import os
import sys

import tensorflow as tf

from gan_model import model
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

    generator_input = model_ops.generator_input(args.samples)
    generated = model_ops.generator(generator_input, hparams)

    saver = tf.train.Saver()
    with tf.Session() as session:
        # Initializing the model using a saved checkpoint.
        saver.restore(session, args.load_checkpoint)

        global_step = session.run(model_ops.global_step)
        print("Load model at step %d." % global_step)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        for sample in session.run(generated):
            print(sample[0])


if __name__ == "__main__":
    main(sys.argv[1:])
