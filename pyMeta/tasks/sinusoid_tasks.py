"""
Utility functions to create sinusoid-fitting regression tasks.

The created tasks will be derived from ClassificationTask, and can be aggregated in a TaskDistribution object.
"""

import numpy as np

from pyMeta.core.task import ClassificationTask
from pyMeta.core.task_distribution import TaskDistribution


class SinusoidTask(ClassificationTask):
    def __init__(self, min_amplitude=0.1,
                 max_amplitude=5.0,
                 min_phase=0.0,
                 max_phase=np.pi,
                 min_x=-5.0,
                 max_x=5.0,
                 num_training_samples=10,
                 num_test_samples=100):
        """
        The test set is generated evenly from min_x to max_x.
        """
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude
        self.min_phase = min_phase
        self.max_phase = max_phase
        self.min_x = min_x
        self.max_x = max_x
        self.num_training_samples = num_training_samples
        self.num_test_samples = num_test_samples

        self.reset()

    def reset(self):
        # Generate a new sinusoid training set
        amplitude = np.random.random() * (self.max_amplitude - self.min_amplitude) + self.min_amplitude
        phase = np.random.random() * (self.max_phase - self.min_phase) + self.min_phase

        self.X = np.random.random((self.num_training_samples, 1)) * (self.max_x - self.min_x) + self.min_x
        self.y = amplitude * np.sin(self.X + phase)

        step = (self.max_x - self.min_x) / self.num_test_samples
        self.test_X = np.expand_dims(np.arange(self.min_x, self.max_x+step, step), -1)
        self.test_y = amplitude * np.sin(self.test_X + phase)

    def sample_batch(self, batch_size):
        batch_indices = np.random.choice(range(self.num_training_samples), batch_size, replace=False)
        batch_X = np.asarray([self.X[i, :] for i in batch_indices])
        batch_y = np.asarray([self.y[i] for i in batch_indices])
        return batch_X, batch_y

    def get_train_set(self):
        return self.X, self.y

    def get_test_set(self):
        return self.test_X, self.test_y


def create_sinusoid_task_distribution(min_amplitude=0.1,
                                      max_amplitude=5.0,
                                      min_phase=0.0,
                                      max_phase=np.pi,
                                      min_x=-5.0,
                                      max_x=5.0,
                                      num_training_samples=10,
                                      num_test_samples=100,
                                      num_test_tasks=100,
                                      meta_batch_size=5):
    tasks_list = [SinusoidTask(min_amplitude=min_amplitude,
                               max_amplitude=max_amplitude,
                               min_phase=min_phase,
                               max_phase=max_phase,
                               min_x=min_x,
                               max_x=max_x,
                               num_training_samples=num_training_samples,
                               num_test_samples=num_test_samples)]

    metatrain_task_distribution = TaskDistribution(tasks=tasks_list,
                                                   task_probabilities=[1.0],
                                                   batch_size=meta_batch_size,
                                                   sample_with_replacement=True)

    metaval_task_distribution = TaskDistribution(tasks=tasks_list,
                                                 task_probabilities=[1.0],
                                                 batch_size=meta_batch_size,
                                                 sample_with_replacement=True)

    metatest_task_distribution = TaskDistribution(tasks=tasks_list,
                                                  task_probabilities=[1.0],
                                                  batch_size=meta_batch_size,
                                                  sample_with_replacement=True)

    return metatrain_task_distribution, metaval_task_distribution, metatest_task_distribution
