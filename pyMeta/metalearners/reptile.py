"""
Implementation of the Reptile algorithm for meta-learning
Nichol, Achiam and Schulman, (2018) - https://arxiv.org/abs/1803.02999
The meta-gradient is computed as g = (init_theta - avg_final_theta)
"""

import numpy as np
import tensorflow as tf

from pyMeta.core.meta_learner import GradBasedMetaLearner


class ReptileMetaLearner(GradBasedMetaLearner):
    def __init__(self, model, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), name="ReptileMetaLearner"):
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
        # Return the final weights for later use in the `update' method
        new_vars = [v.numpy() for v in self.model.trainable_variables]
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

        # Apply gradients to the *initial parameters*
        for i in range(len(self.current_initial_parameters)):
            self.model.trainable_variables[i].assign( self.current_initial_parameters[i] )

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Set the new initial parameters
        self.current_initial_parameters = [v.numpy() for v in self.model.trainable_variables]

