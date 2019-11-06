"""
Utility functions to create permuted-mnist Tasks, sampling a new permutation on each task reset.

The created tasks will be derived from ClassificationTask, and can be aggregated in a TaskDistribution object.
"""

import numpy as np
import tensorflow as tf
import pickle

from pyMeta.core.task import ClassificationTask
from pyMeta.core.task_distribution import TaskDistribution


cifar100_trainX = []
cifar100_trainY = []
cifar100_testX = []
cifar100_testY = []


def create_cifar100_task_distribution(num_training_samples_per_class=-1,
                                      num_test_samples_per_class=-1,
                                      num_training_classes=10,
                                      meta_train_test_split=0.7,
                                      meta_batch_size=5):
    """
    Returns a TaskDistribution that, on each reset, samples a different set of CIFAR-100 classes.

    Note that the first time this function is called on a new system, it will download the CIFAR-100 dataset, which
    may take some time (usually less than 5 minutes).

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
    meta_train_test_split : float
        Proportion of classes to use for the meta-training set. E.g., split=0.7 means int(0.7*100)=70 classes will
        be used for meta-training, while 100-70=30 classes will be used for meta-testing.
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

    global cifar100_trainX
    global cifar100_trainY
    global cifar100_testX
    global cifar100_testY

    ((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

    all_x = np.concatenate((train_data, eval_data), axis=0)
    all_y = np.concatenate((train_labels, eval_labels), axis=0)

    split_class = int(meta_train_test_split * 100)

    meta_train_classes = list(range(split_class))
    meta_test_classes  = list(range(split_class, 100))

    meta_train_indices = []
    for c in meta_train_classes:
        c_indices = np.where(all_y == c)[0]
        meta_train_indices.extend(c_indices)

    meta_test_indices = []
    for c in meta_test_classes:
        c_indices = np.where(all_y == c)[0]
        meta_test_indices.extend(c_indices)

    # TODO: subtract mean of train images (over axis=0) from both trainX and testX

    """
    from copy import copy
    import cv2
    old_x = copy(all_x)
    all_x = np.ones([old_x.shape[0], 224, 224, 3], dtype=np.float32)
    for i in range(old_x.shape[0]):
        all_x[i,:,:,:] = cv2.resize(old_x[i,:,:,:], (224, 224))
    """

    cifar100_trainX = all_x[meta_train_indices, :].astype(np.float32) / 255.0
    cifar100_trainY = np.squeeze(all_y[meta_train_indices]).astype(np.int64)
    cifar100_testX = all_x[meta_test_indices, :].astype(np.float32) / 255.0
    cifar100_testY = np.squeeze(all_y[meta_test_indices]).astype(np.int64)

    metatrain_tasks_list = [ClassificationTask(cifar100_trainX,
                                               cifar100_trainY,
                                               num_training_samples_per_class,
                                               num_test_samples_per_class,
                                               num_training_classes,
                                               split_train_test=-1)] # defaults to num_train / (num_train+num_test)
    metatest_tasks_list = [ClassificationTask(cifar100_testX,
                                              cifar100_testY,
                                              num_training_samples_per_class,
                                              num_test_samples_per_class,
                                              num_training_classes,
                                              split_train_test=-1)]

    metatrain_task_distribution = TaskDistribution(tasks=metatrain_tasks_list,
                                                   task_probabilities=[1.0],
                                                   batch_size=meta_batch_size,
                                                   sample_with_replacement=True)

    metatest_task_distribution = TaskDistribution(tasks=metatest_tasks_list,
                                                  task_probabilities=[1.0],
                                                  batch_size=meta_batch_size,
                                                  sample_with_replacement=True)

    # TODO: split into validation and test!
    return metatrain_task_distribution, metatest_task_distribution, metatest_task_distribution
