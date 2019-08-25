"""
Implements the TaskDistribution and TaskAsSequenceOfTasksDistribution classes.
"""

import numpy as np

from copy import deepcopy


class TaskDistribution:
    def __init__(self, tasks, task_probabilities, batch_size=1, sample_with_replacement=True):
        """
        Wrapper around tasks for use in meta-learning.
        TaskDistribution allows the definition of a distribution over Tasks from which batches of tasks can be easily
        sampled.

        Note : many problems in which a single Task can generate multiple different problems (e.g., a classifier which
        is a subset of a larger dataset) can be fit using a TaskDistribution with a list of a single element, the task
        prototype, allowing for `sample_with_replacement'. If `sample_with_replacement' is True, the same task can be
        returned multiple time within the same batch of tasks, as *deepcopy*-ed copies.

        tasks: list
            List of tasks to sample from. Each task can itself sample different parameters each time it is reset,
            or can even be another TaskDistribution.
        task_probabilities: list
            List of task probabilities used for sampling tasks. The list is normalized by the sum of its elements,
            in case the supplied list does not sum to 1.
        batch_size: int
            Default size of batch size of tasks to sample.
        sample_with_replacement: bool
            If with_replacement=False, tasks can only be sampled once in a meta-batch. This allows for faster and
            more efficient treatment of the tasks, but it would not work well in cases where each task itself offers
            many variants (e.g., via reset-sampled parameters).
            If with_replacement=True, tasks are allowed to be picked multiple times. Each sample is *deepcopied* from
            the original Task object, and hard reset, so that it can sample different parameters (e.g., in the sinusoid
            regression example.)
            WARNING: if TaskDistribution objects are compounded, deepcopying them may become expensive. It is better
            to use a single TaskDistribution, and aggregate all lists together, with the appropriate probabilities.
        """
        self.tasks = tasks

        sum_probabilities = sum(task_probabilities)
        self.task_probabilities = [tp/sum_probabilities for tp in task_probabilities]

        self.batch_size = batch_size
        self.sample_with_replacement = sample_with_replacement

        self.reset()

    def reset(self):
        return

    def set_task_probabilities(self, task_probabilities):
        self.task_probabilities = task_probabilities

    def sample_batch(self, batch_size=None):
        """
        Sample `batch_size' tasks from the list of tasks. This performs a deep-copy on each task and resets them to
        generate new tasks from each distribution.
        """
        if batch_size is None:
            batch_size = self.batch_size

        if self.sample_with_replacement:
            tmp_batch = np.random.choice(self.tasks, batch_size, replace=True)
            batch = []

            included_tasks = set()
            for b in tmp_batch:
                if b in included_tasks:
                    newb = deepcopy(b)
                    newb.reset()
                    batch.append(newb)
                else:
                    b.reset()
                    batch.append(b)
                    included_tasks.add(b)
        else:
            batch = np.random.choice(self.tasks, batch_size, replace=False)

        return batch

    def get_num_tasks(self):
        return len(self.tasks)

    def get_task_by_index(self, index):
        assert (index >= 0 and index < len(self.tasks)), "INVALID TASK INDEX"

        return self.tasks[index]


class TaskAsSequenceOfTasksDistribution(TaskDistribution):
    def __init__(self, task_sequence_object, batch_size):
        """
        Equivalent to TaskDistribution, and can only be used with task objects of type `TaskAsSequenceOfTasks'.
        *This object samples `batch_size' sequences from a distribution defined by the TaskAsSequenceOfTasks
        `task_sequence_object'.*

        When using *only* Tasks consisting of sequences of sub-tasks, this implementation is more efficient than
        TaskDistribution. However, it can't handle mixtures of task sequences and other tasks, unless the task sequence
        object is extended to allow for multiple-length sequences.

        task_sequence_object: TaskAsSequenceOfTasks
            Task object of type TaskAsSequenceOfTasks
        batch_size: int
            For this implementation, the batch_size (number of tasks-sequences to sample at each meta-iteration)
            must be known in advance.
        """
        self.task_sequences_batch = [deepcopy(task_sequence_object) for _ in range(batch_size)]
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        for i in range(self.batch_size):
            self.task_sequences_batch[i].reset()

    def sample_batch(self, batch_size=None):
        """
        Sample a task sequence from each generator (total: `self.batch_size' tasks returned).
        The `batch_size' argument is ignored, and is only included for compatiblity with TaskDistribution.
        """
        for i in range(self.batch_size):
            self.task_sequences_batch[i].reset()

        return self.task_sequences_batch

    def get_num_tasks(self):
        return self.batch_size

    def get_task_by_index(self, index):
        assert (index >= -1 and index < self.batch_size), "INVALID TASK INDEX"

        return self.task_sequences_batch[index]
