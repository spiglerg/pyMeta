"""
Utility functions to create Tasks from the CUB-200 dataset.

The created tasks will be derived from ClassificationTask, and can be aggregated in a TaskDistribution object.
"""

import os
import pickle
import cv2
import numpy as np

from pyMeta.core.task import ClassificationTask, ClassificationTaskFromFiles
from pyMeta.core.task_distribution import TaskDistribution


# All data is pre-loaded in memory. This takes ~5GB if I recall correctly.
cub200_trainX = []
cub200_trainY = []

cub200_valX = []
cub200_valY = []

cub200_testX = []
cub200_testY = []



# TODO: allow for a custom train/test ratio split for each class!
def create_cub200_task_distribution(path_to_pkl,
                                          num_training_samples_per_class=10,
                                          num_test_samples_per_class=15,
                                          num_training_classes=20,
                                          meta_batch_size=5):
    """
    Returns a TaskDistribution that, on each reset, samples a different set of Mini-ImageNet classes.

    *** Data is loaded from a special pickle file. ***

    Arguments:
    path_to_pkl: string
        Path to the pkl wrapped Mini-ImageNet dataset. This can be generated from the standard dataset using the
        supplied make_miniimagenet_dataset.py script.
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

    global cub200_trainX
    global cub200_trainY

    global cub200_valX
    global cub200_valY

    global cub200_testX
    global cub200_testY


    with open(path_to_pkl, 'rb') as f:
        d = pickle.load(f)
        cub200_X, cub200_Y = d['dataset']

    cub200_X = cub200_X.astype(np.float32) / 255.0
    cub200_X = (cub200_X - np.asarray((0.4914, 0.4822, 0.4465))) / np.asarray((0.2023, 0.1994, 0.2010))

    #
    # TODO
    # random horiz flip + normalize by: 
    #        transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                             (0.2023, 0.1994, 0.2010))   (mean, std)



    #np.random.seed(0)
    # TODO: shuffle allocation of class indices to train/val/test
    num_train = 100
    num_val = 50
    num_test = 50

    classes = list(set(cub200_Y))
    train_classes = classes[:num_train]
    val_classes = classes[num_train:(num_train+num_val)]
    test_classes = classes[(num_train+num_val):]

    train_indices = []
    val_indices = []
    test_indices = []

    for i in range(len(cub200_Y)):
        if cub200_Y[i] in train_classes:
            train_indices.append(i)
        elif cub200_Y[i] in val_classes:
            val_indices.append(i)
        elif cub200_Y[i] in test_classes:
            test_indices.append(i)

    cub200_trainX = cub200_X[train_indices]
    cub200_trainY = cub200_Y[train_indices]

    cub200_valX = cub200_X[val_indices]
    cub200_valY = cub200_Y[val_indices]

    cub200_testX = cub200_X[test_indices]
    cub200_testY = cub200_Y[test_indices]


    train_tasks_list = [ClassificationTask(cub200_trainX,
                                           cub200_trainY,
                                           num_training_samples_per_class,
                                           num_test_samples_per_class,
                                           num_training_classes,
                                           split_train_test=0.5)]

    # TODO: NOTE: HACK -- validation and test tasks use a fixed number of test-set samples, instead of the supplied
    # ones. This is because in MAML/FOMAML the test set is used to compute the meta-gradient, and a small number of
    # samples is used (in the philosophy of few-shot learning, where only few samples are available).
    # However, in this case we wish to use a few more test-samples to better estimate the accuracy of the model on the validation
    # and test tasks!
    num_test_samples_per_class = 20
    validation_tasks_list = [ClassificationTask(cub200_valX,
                                                cub200_valY,
                                                num_training_samples_per_class,
                                                num_test_samples_per_class,
                                                num_training_classes,
                                                split_train_test=0.5)]

    test_tasks_list = [ClassificationTask(cub200_valX,
                                          cub200_valY,
                                          num_training_samples_per_class,
                                          num_test_samples_per_class,
                                          num_training_classes,
                                          split_train_test=0.5)]

    metatrain_task_distribution = TaskDistribution(tasks=train_tasks_list,
                                                   task_probabilities=[1.0],
                                                   batch_size=meta_batch_size,
                                                   sample_with_replacement=True,
                                                   use_classes_only_once=True)

    metaval_task_distribution = TaskDistribution(tasks=validation_tasks_list,
                                                 task_probabilities=[1.0],
                                                 batch_size=meta_batch_size,
                                                 sample_with_replacement=True,
                                                   use_classes_only_once=True)

    metatest_task_distribution = TaskDistribution(tasks=test_tasks_list,
                                                  task_probabilities=[1.0],
                                                  batch_size=meta_batch_size,
                                                  sample_with_replacement=True,
                                                   use_classes_only_once=True)

    return metatrain_task_distribution, metaval_task_distribution, metatest_task_distribution
