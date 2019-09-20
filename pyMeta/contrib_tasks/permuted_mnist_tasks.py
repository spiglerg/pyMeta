"""
Utility functions to create permuted-mnist Tasks, sampling a new permutation on each task reset.

The created tasks will be derived from ClassificationTask, and can be aggregated in a TaskDistribution object.
"""

import numpy as np
import tensorflow as tf
import pickle

from pyMeta.core.task import ClassificationTask
from pyMeta.core.task_distribution import TaskDistribution


mnist_X = []
mnist_y = []

class PermutedMNIST_Task(ClassificationTask):
    def __init__(self, X, y, num_training_samples_per_class=-1, num_test_samples_per_class=-1, num_training_classes=-1, split_train_test=0.8):
        super().__init__(X, y, num_training_samples_per_class, num_test_samples_per_class, num_training_classes, split_train_test)

    def reset(self):
        super().reset()

        # Generate a new permutation
        self.permutation = np.random.permutation(784)

    def sample_batch(self, batch_size):
        bx, by = super().sample_batch(batch_size)
        bx = bx[:, self.permutation]
        return bx, by

    def get_train_set(self):
        x, y = super().get_train_set()
        x = x[:, self.permutation]
        return x, y

    def get_test_set(self):
        x, y = super().get_test_set()
        x = x[:, self.permutation]
        return x, y


def make_permuted_mnist_mlp_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(400, input_shape=[784]))
    model.add(tf.keras.layers.Dense(400, input_shape=[784]))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model


def create_permuted_mnist_task_distribution(num_training_samples_per_class=-1,
                                            num_test_samples_per_class=-1,
                                            num_training_classes=10,
                                            meta_batch_size=5):
    """
    Returns a TaskDistribution that, on each reset, samples a different set of omniglot characters.

    Arguments:
    num_training_samples_per_class : int
        If -1, sample from the whole dataset. If >=1, the dataset will re-sample num_training_samples_per_class
        for each class at each reset, and sample minibatches exclusively from them, until the next reset.
        This is useful for, e.g., k-shot classification.
    num_test_samples_per_class : int
        Same as `num_training_samples_per_class'. Used to generate test sets for tasks on reset().
    num_training_classes : int
        If -1, use all the classes in `y'. If >=1, the dataset will re-sample `num_training_classes' at
        each reset, and sample minibatches exclusively from them, until the next reset.
    meta_batch_size : int
        Default size of the meta batch size.

    Returns:
    metatrain_task_distribution : TaskDistribution
        TaskDistribution object for use during training
    metaval_task_distribution : TaskDistribution
        TaskDistribution object for use during model validation
    metatest_task_distribution : TaskDistribution
        TaskDistribution object for use during testing
    """

    global mnist_X
    global mnist_y

    ((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

    mnist_X = np.reshape(np.concatenate((train_data, eval_data), axis=0), [-1, 28*28]).astype(np.float32) / 255.0
    mnist_y = np.concatenate((train_labels, eval_labels), axis=0).astype(np.int64)

    tasks_list = [PermutedMNIST_Task(mnist_X,
                                     mnist_y,
                                     num_training_samples_per_class,
                                     num_test_samples_per_class,
                                     num_training_classes,
                                     split_train_test=0.8)]

    task_distribution = TaskDistribution(tasks=tasks_list,
                                         task_probabilities=[1.0],
                                         batch_size=meta_batch_size,
                                         sample_with_replacement=True)

    return task_distribution, task_distribution, task_distribution
