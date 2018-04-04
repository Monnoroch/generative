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

    experiment = Experiment.from_checkpoint(args.load_checkpoint)
    hparams = experiment.load_hparams(model.ModelParams)

    # Create the model.
    dataset = model.DatasetParams(args)
    model_ops = model.GanNormalModel(None, None, model.TrainingParams(None, training=False))

    with tf.name_scope("global_step_tools"):
        global_step_op = tf.Variable(0, name="global_step", trainable=False)

    real_samples = model_ops.data_batch(dataset, args.samples)

    saver = tf.train.Saver()
    with tf.Session() as session:
        # Initializing the model using a saved checkpoint.
        saver.restore(session, args.load_checkpoint)

        global_step = session.run(global_step_op)
        print("Load model at step %d." % global_step)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        for sample in session.run(real_samples):
            print(sample[0])


if __name__ == "__main__":
    main(sys.argv[1:])
