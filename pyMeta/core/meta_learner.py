"""
Base interface for meta-learners based on gradients.

The interface is just a tentative suggestion, as different algorithms may have very different needs.
"""

import tensorflow as tf

from pyMeta.core.task import Task


class GradBasedMetaLearner:
    """
    In general, meta-learners objects should be created before the tf.keras.Model that they wrap is compiled,
    in case losses or regularizers need to be added.

    This meta-learner should be used as follows:
    + Instantiate object
    [Compile the wrapped model]
    + Initialize object
        metalearner.initialize()
    + For each meta-learning iteration:
        + Go through each task in the meta-batch
        + Tell the meta-learner a new task is starting
            metalearner.task_begin(task)
        + Train the task
        + Tell the meta-learner the task has finished training, and collect the returned values
            results.append(metalearner.task_end(task))
    + Perform a meta-update
        metalearner.update(results)

    optimizer: tf.train.Optimizer objects
        The optimizer to use for meta-learning updates (outer loop).
    model: tf.keras.Model
        The model to meta-optimize.
    """

    def initialize(self, session):
        """
        This method should be called after the wrapped model is compiled.
        """
        pass

    def task_begin(self, task=None, **kwargs):
        """
        Method to be called before training on each meta-batch task.
        """
        if task is not None and not isinstance(task, Task):
            print("ERROR: task_begin requires a `Task' object as argument")

    def task_end(self, task=None, **kwargs):
        """
        Method to call after training on each meta-batch task; possibly return relevant information for the
        meta-learner to use for the meta-updates.
        """
        pass

    def update(self, list_of_metabatch_data, **kwargs):
        """
        Main (outer-loop) training operation. Call after training on a whole meta-batch, after each meta-iteration
        has finished.
        """
        pass
