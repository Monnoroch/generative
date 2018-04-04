import tensorflow as tf

class TrainingLoopParams(object):
    """
    All parameters for the training loop that should be configured from command line should go here.
    """
    def __init__(self, args):
        self.max_steps = args.max_steps
        self.steps_between_summary_exports = args.steps_between_summary_exports

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--max_steps", type=int, default=1000,
            help="The maximum number of steps to train training for")
        parser.add_argument("--steps_between_summary_exports", type=int, default=1,
            help="The number of training steps betweed summary exports")

def training_loop(params, experiment, summaries_op, callback, checkpoint=None):
    """
    Training loop function.

    Does all the bookkeeping for training the model and calls the callback for training iteration.

    Arguments:
    - params - training loop params.
    - experiment - the experiment object to use.
    - summaries_op - a summary TF op.
    - callback - the callback to be called on each iteration. Should recieve a session object and a global step tensor.
    - checkpoint - an optional checkpoint path.
    """
    with tf.name_scope("global_step_tools"):
        global_step_op = tf.Variable(0, name="global_step", trainable=False)
        increment_global_step_op = tf.assign_add(global_step_op, 1)

    saver = tf.train.Saver()
    with tf.Session() as session:
        # Initialize the model either using a saved checkpoint or a ranrom initializer.
        if checkpoint:
            saver.restore(session, checkpoint)
        else:
            session.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(experiment.summaries_dir(), session.graph)

        # The main training loop.
        global_step = session.run(global_step_op)
        for _ in range(params.max_steps):
            callback(session, global_step)
            session.run(increment_global_step_op)
            global_step = session.run(global_step_op)
            # Export all summaries to tensorboard.
            if global_step % params.steps_between_summary_exports == 0:
                summary_writer.add_summary(session.run(summaries_op), global_step)

        # Save experiment data.
        saver.save(session, experiment.checkpoint(global_step))
