"""
Implementation of the First-Order MAML (FOMAML) algorithm for meta-learning.
For details, see https://arxiv.org/abs/1909.04170 .
"""

import numpy as np
import tensorflow as tf

from pyMeta.core.task import TaskAsSequenceOfTasks
from pyMeta.metalearners.fomaml import FOMAMLMetaLearner


class SeqFOMAMLMetaLearner(FOMAMLMetaLearner):
    def __init__(self, model, optimizer=tf.train.AdamOptimizer(learning_rate=0.001), name="SeqFOMAMLMetaLearner"):
        """
        Extension of FOMAML to learn to avoid catastrophic forgetting by training on sequences of tasks.
        The base gradient is the expected final (test) gradient, averaged across tasks in a sequence. Auxliary
        gradients are added to make it work as desired. `loss_after_each_task' performs best, and works by adding
        gradients for each task computed at the final parameters after training on the corresponding task.

        See FOMAMLMetaLearner for the detailed documentation.
        """
        super().__init__(model=model, optimizer=optimizer, name=name)

    def task_end(self, task=None, **kwargs):
        """
        Method to call after training on each meta-batch task; possibly return relevant information for the
        meta-learner to use for the meta-updates.
        """
        assert task is not None, "FOMAML needs a `task' argument on .task_end to compute the relevant data."
        assert isinstance(task, TaskAsSequenceOfTasks), "Tas must be a TaskAsSequenceofTasks object."
        # Compute the gradient of the TEST LOSS evaluated at the final parameters. E.g., mean gradients
        # over a batch of test data

        # Default: evaluate the performance of all tasks in the sequence at the final weights (current weights
        # on task end). Return the average gradient over the test losses of all the tasks in the sequence.
        grads = []
        for i in range(task.get_sequence_length()):
            grads.append(self._gradients_for_task(task.get_task_by_index(i)))

        ret_grads = []
        for gs in zip(*grads):
            ret_grads.append(np.mean(gs, axis=0))

        # Auxiliary losses
        if 'results' in kwargs:
            results = kwargs['results']

        # TODO: from auxiliary losses code
        if ('seq_fomaml_loss' not in kwargs) or kwargs['seq_fomaml_loss'] == 'plain':
            # 1/n * sum_i[ L_i(phi^n) ]
            aux_grads = []
        else:
            aux_grads = []

            if kwargs['seq_fomaml_loss'] == 'loss_after_each_task':
                # Adds extra term  L_i(phi^i) term

                for i in range(task.get_sequence_length()):
                    # Compute loss of task `i' at its previous final weights
                    self.session.run(self._assign_op, feed_dict=dict(zip(self._placeholders,
                                                                         results['weights_'+str(i)])))
                    aux_grads.append(self._gradients_for_task(task.get_task_by_index(i)))

            elif kwargs['seq_fomaml_loss'] == 'sampled_loss_after_each_task':
                # Tests on a randomly sampled previous task, after each task: adds extra term L_{j ~ U([1,i])}(phi^i)

                for i in range(0, task.get_sequence_length()):
                    self.session.run(self._assign_op, feed_dict=dict(zip(self._placeholders,
                                                                         results['weights_'+str(i)])))
                    j = np.random.randint(0, i+1)
                    aux_grads.append(self._gradients_for_task(task.get_task_by_index(j)))

            elif kwargs['seq_fomaml_loss'] == 'all_losses_after_each_task':
                # Tests on all previous tasks, after each task: adds extra term sum_j<=i L_j(phi^i) / (n+1-j)
                # TODO: test - new untested implementation

                # No need to compute the losses at the end of training t=task.get_sequence_length()-1,
                # as those are collected above
                for i in range(task.get_sequence_length()-1):
                    for j in range(i+1):
                        self.session.run(self._assign_op, feed_dict=dict(zip(self._placeholders,
                                                                             results['weights_'+str(i)])))
                        newg = self._gradients_for_task(task.get_task_by_index(j))
                        for k in range(len(newg)):
                            newg[k] /= (task.get_sequence_length()+1-j)
                        aux_grads.append(newg)

                # Append the last gradients for each task, for i=n
                # (computed in grads, but need to rescale them by n+1-j)
                # for j in range(task.get_sequence_length()):
                #     newg = [g/((task.get_sequence_length()+1-j)) for g in grads]
                #     aux_grads.append(newg)

            ret_grads_1 = []
            ret_grads_2 = []
            for gs in zip(*grads):
                ret_grads_1.append(np.mean(gs, axis=0))
            for ags in zip(*aux_grads):
                ret_grads_2.append(np.mean(ags, axis=0))
            ret_grads = [ret_grads_1[i]+ret_grads_2[i] for i in range(len(ret_grads_1))]

        return ret_grads
