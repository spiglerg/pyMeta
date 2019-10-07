"""
Implementation of the Reptile algorithm for meta-learning
Nichol, Achiam and Schulman, (2018) - https://arxiv.org/abs/1803.02999
The meta-gradient is computed as g = (init_theta - avg_final_theta)
"""

import numpy as np
import tensorflow as tf

from pyMeta.core.meta_learner import GradBasedMetaLearner


class ReptileMetaLearner(GradBasedMetaLearner):
    def __init__(self, model, optimizer=tf.train.AdamOptimizer(learning_rate=0.001), name="ReptileMetaLearner"):
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

        # Update op to change the current initial parameters (weights initialization)
        # From OpenAI Reptile implementation
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
        # Reset the model to the current weights initialization
        self.session.run(self._assign_op, feed_dict=dict(zip(self._placeholders, self.current_initial_parameters)))

    def task_end(self, task=None, **kwargs):
        """
        Method to call after training on each meta-batch task; possibly return relevant information for the
        meta-learner to use for the meta-updates.
        """
        # Return the final weights for later use in the `update' method
        new_vars = self.session.run(self.model.trainable_variables)
        return new_vars

    def update(self, metabatch_results, **kwargs):
        """
        Main (outer-loop) training operation. Call after training on a whole meta-batch, after each meta-iteration
        has finished.

        metabatch_results contains a list of the final parameters of the network, for each task in the meta-batch
        """
        # Perform a Reptile update for the outer meta-training iteration

        # Compute mean of the final weights for each task in the meta-batch
        avg_final = []
        for variables in zip(*metabatch_results):
            avg_final.append(np.mean(variables, axis=0))

        # Move current initial weights towards the avg final weights with step size `meta_learning_rate'
        # Manual implementation of SGD:
        # self.current_initial_parameters = [cur - self.meta_learning_rate*(cur-new)
        #                                    for cur, new in zip(self.current_initial_parameters, avg_final)]

        # Better implementation, using Tensorflow optimizers
        grads = []
        for cur, new in zip(self.current_initial_parameters, avg_final):
            grads.append(cur-new)

        # Apply gradients to the initial parameters
        self.session.run(self._assign_op, feed_dict=dict(zip(self._placeholders, self.current_initial_parameters)))
        self.session.run(self.apply_metagradients, feed_dict=dict(zip(self._gradients_placeholders, grads)))

        self.current_initial_parameters = self.session.run(self.model.trainable_variables)
