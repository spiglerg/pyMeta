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


class FOMAMLMetaLearner(GradBasedMetaLearner):
    def __init__(self, model, optimizer=tf.train.AdamOptimizer(learning_rate=0.001), name="FOMAMLMetaLearner"):
        """
        In general, meta-learners objects should be created before tf.global_variables_initializer() is called, in
        case variables have to be created.

        This meta-learner should be used as follows:
        + Instantiate object (before calling tf.global_variables_initializer() )
        + Initialize object, after tf.global_variables_initializer()
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

        self.target_placeholder = None

        # Update op to change the current initial parameters (weights initialization)
        self._placeholders = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape())
                              for v in self.model.trainable_variables]
        assigns = [tf.assign(v, p) for v, p in zip(self.model.trainable_variables, self._placeholders)]
        self._assign_op = tf.group(*assigns)

        self._gradients_placeholders = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape())
                                        for v in self.model.trainable_variables]
        self.apply_metagradients = self.optimizer.apply_gradients(zip(self._gradients_placeholders,
                                                                      self.model.trainable_variables))
        # clipped_grads = [(tf.clip_by_norm(grad, 10), var) for grad, var in zip(self._gradients_placeholders,
        #                                                                        self.model.trainable_variables)]
        # self.apply_metagradients = self.optimizer.apply_gradients(clipped_grads)

    def _gradients_for_task(self, t):
        """
        Hacky utility function to compute gradients for a given batch of data on a Keras model.
        For the moment, this only works for Keras models with a single input.
        """
        # TODO: this should work in general, but it will require adjustments for RL tasks!
        batch_x, batch_y = t.get_test_set()

        if self.target_placeholder is None:
            # Only build the graph on the first time this function is called
            # `model' is run through self.model.inputs[0] to self.model.output
            self.target_placeholder = tf.placeholder(tf.float32, [None, ] + list(batch_y.shape[1:]),
                                                     name="fomaml_placeholder")

            self.loss = tf.reduce_mean(self.model.loss(self.target_placeholder, self.model.output))
            self.gradients_ = tf.gradients(self.loss, self.model.trainable_variables)

        gradients = self.session.run(self.gradients_,
                                     {self.model.inputs[0]: batch_x, self.target_placeholder: batch_y})
        return gradients

    def initialize(self, session):
        """
        This method should be called after tf.global_variables_initializer().
        """
        self.session = session
        self.current_initial_parameters = self.session.run(self.model.trainable_variables)

    def task_begin(self, task=None, **kwargs):
        """
        Method to be called before training on each meta-batch task
        """
        super().task_begin(task=task)

        # Reset the model to the current weights initialization
        self.session.run(self._assign_op, feed_dict=dict(zip(self._placeholders, self.current_initial_parameters)))

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

        # Apply gradients to the initial parameters
        self.session.run(self._assign_op, feed_dict=dict(zip(self._placeholders, self.current_initial_parameters)))
        self.session.run(self.apply_metagradients, feed_dict=dict(zip(self._gradients_placeholders, avg_final_grads)))

        self.current_initial_parameters = self.session.run(self.model.trainable_variables)
