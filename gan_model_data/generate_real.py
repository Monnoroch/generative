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
    parser.add_argument("--input_mean", type=float, default=[], help="The mean of the input dataset", action="append")
    parser.add_argument("--input_stddev", type=float, default=[], help="The standard deviation of the input dataset", action="append")
    args = parser.parse_args(args)
    # Default input mean and stddev.
    if not args.input_mean:
        args.input_mean.append(15.)
    if not args.input_stddev:
        args.input_stddev.append(7.)
    if len(args.input_mean) != len(args.input_stddev):
        print("There must be the same number of input means and standard deviations.")
        sys.exit(1)

    args.experiment_dir = os.path.dirname(os.path.dirname(os.path.dirname(args.load_checkpoint)))

    # Create the model.
    hparams = model.ModelParams(load_hparams(args))
    dataset = model.DatasetParams(args)
    model_ops = model.GanNormalModel(None, None, model.TrainingParams(None, training=False))

    real_samples = model_ops.data_batch(dataset, args.samples)

    saver = tf.train.Saver()
    with tf.Session() as session:
        # Initializing the model using a saved checkpoint.
        saver.restore(session, args.load_checkpoint)

        global_step = session.run(model_ops.global_step)
        print("Load model at step %d." % global_step)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        for sample in session.run(real_samples):
            print(sample[0])


if __name__ == "__main__":
    main(sys.argv[1:])
