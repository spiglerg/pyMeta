"""
Implementation of the First-Order MAML (FOMAML) algorithm for meta-learning.
Finn, Abbeel and Levine, (2017) - https://arxiv.org/abs/1703.03400
The meta-gradient is computed as g = \frac{\partial}{\partial \theta} L_{test}(\theta),
evaluated at \theta=\theta_{final}  (i.e., gradient wrt to the final parameters after inner-loop optimization)
"""

import numpy as np
import tensorflow as tf

from pyMeta.core.meta_learner import GradBasedMetaLearner
from pyMeta.core.task import TaskAsSequenceOfTasks





@tf.function
def grads_on_batch(model, batch_X, batch_y):
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(model.loss(batch_y, model(batch_X, training=True)))
    grads = tape.gradient(loss, model.trainable_variables)
    return grads

class FOMAMLMetaLearner(GradBasedMetaLearner):
    def __init__(self, model, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), name="FOMAMLMetaLearner"):
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
        self.model = model
        self.optimizer = optimizer

    def _gradients_for_task(self, t):
        """
        Hacky utility function to compute gradients for a given batch of data on a Keras model.
        For the moment, this only works for Keras models with a single input.
        """
        # TODO: this should work in general, but it will require adjustments for RL tasks!
        batch_x, batch_y = t.get_test_set()

        gradients = grads_on_batch(self.model, batch_x, batch_y)
        gradients = [g.numpy() for g in gradients]

        return gradients

    def initialize(self):
        """
        This method should be called after the wrapped model is compiled.
        """
        self.current_initial_parameters = [v.numpy() for v in self.model.trainable_variables]

    def task_begin(self, task=None, **kwargs):
        """
        Method to be called before training on each meta-batch task
        """
        super().task_begin(task=task)

        # Reset the model to the current weights initialization
        for i in range(len(self.current_initial_parameters)):
            self.model.trainable_variables[i].assign( self.current_initial_parameters[i] )

    def task_end(self, task=None, **kwargs):
        """
        Method to call after training on each meta-batch task; possibly return relevant information for the
        meta-learner to use for the meta-updates.
        """
        assert task is not None, "FOMAML needs a `task' argument on .task_end to compute the data it needs."

        # Compute the gradient of the TEST LOSS evaluated at the final parameters. E.g., mean gradients
        # over a batch of test data

        if isinstance(task, TaskAsSequenceOfTasks):
            # Default: evaluate performance on the last task only
            ret_grads = self._gradients_for_task(task.get_task_by_index(-1))
        else:
            ret_grads = self._gradients_for_task(task)

        return ret_grads

    def update(self, list_of_final_gradients, **kwargs):
        """
        Main (outer-loop) training operation. Call after training on a whole meta-batch, after each meta-iteration
        has finished.
        """
        # Perform a FOMAML update for the outer meta-training iteration. Take the expected test gradient
        # L'(\tilde{phi}) over tasks in the meta-batch and perform a single step of gradient descent.
        avg_final_grads = []
        for grads in zip(*list_of_final_gradients):
            avg_final_grads.append(np.mean(grads, axis=0))

        # Apply gradients to the *initial parameters*
        for i in range(len(self.current_initial_parameters)):
            self.model.trainable_variables[i].assign( self.current_initial_parameters[i] )

        self.optimizer.apply_gradients(zip(avg_final_grads, self.model.trainable_variables))

        # Set the new initial parameters
        self.current_initial_parameters = [v.numpy() for v in self.model.trainable_variables]
