"""
Utility functions to create Tasks from the Omniglot dataset.

The created tasks will be derived from ClassificationTask, and can be aggregated in a TaskDistribution object.
"""

import numpy as np
import pickle

from pyMeta.core.task import ClassificationTask
from pyMeta.core.task_distribution import TaskDistribution

charomniglot_trainX = []
charomniglot_trainY = []
charomniglot_testX = []
charomniglot_testY = []


# TODO: allow for a custom train/test ratio split for each class!
def create_omniglot_allcharacters_task_distribution(path_to_pkl,
                                                    num_training_samples_per_class=10,
                                                    num_test_samples_per_class=-1,
                                                    num_training_classes=20,
                                                    meta_batch_size=5):
    """
    Returns a TaskDistribution that, on each reset, samples a different set of omniglot characters.

    Arguments:
    path_to_pkl: string
        Path to the pkl wrapped Omniglot dataset. This can be generated from the standard dataset using the supplied
        make_omniglot_dataset.py script.
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

    with open(path_to_pkl, 'rb') as f:
        d = pickle.load(f)
        trainX_ = d['trainX']
        trainY_ = d['trainY']
        testX_ = d['testX']
        testY_ = d['testY']
    trainX_.extend(testX_)
    trainY_.extend(testY_)

    global charomniglot_trainX
    global charomniglot_trainY
    global charomniglot_testX
    global charomniglot_testY

    cutoff = 36
    charomniglot_trainX = trainX_[:cutoff]
    charomniglot_trainY = trainY_[:cutoff]
    charomniglot_testX = trainX_[cutoff:]
    charomniglot_testY = trainY_[cutoff:]

    # Create a single large dataset with all characters, each for train and test, and rename the targets appropriately
    trX = []
    trY = []
    teX = []
    teY = []

    cur_label_start = 0
    for alphabet_i in range(len(charomniglot_trainY)):
        charomniglot_trainY[alphabet_i] += cur_label_start
        trX.extend(charomniglot_trainX[alphabet_i])
        trY.extend(charomniglot_trainY[alphabet_i])
        cur_label_start += len(set(charomniglot_trainY[alphabet_i]))

    cur_label_start = 0
    for alphabet_i in range(len(charomniglot_testY)):
        charomniglot_testY[alphabet_i] += cur_label_start
        teX.extend(charomniglot_testX[alphabet_i])
        teY.extend(charomniglot_testY[alphabet_i])
        cur_label_start += len(set(charomniglot_testY[alphabet_i]))

    trX = np.asarray(trX, dtype=np.float32) / 255.0
    trY = np.asarray(trY, dtype=np.float32)
    teX = np.asarray(teX, dtype=np.float32) / 255.0
    teY = np.asarray(teY, dtype=np.float32)

    charomniglot_trainX = trX
    charomniglot_testX = teX
    charomniglot_trainY = trY
    charomniglot_testY = teY

    print('Loaded ', len(trY), 'training classes and ', len(teY), 'test classes.')

    metatrain_tasks_list = [ClassificationTask(charomniglot_trainX,
                                               charomniglot_trainY,
                                               num_training_samples_per_class,
                                               num_test_samples_per_class,
                                               num_training_classes,
                                               split_train_test=-1)] # defaults to num_train / (num_train+num_test)
    metatest_tasks_list = [ClassificationTask(charomniglot_testX,
                                              charomniglot_testY,
                                              num_training_samples_per_class,
                                              num_test_samples_per_class,
                                              num_training_classes,
                                              split_train_test=-1)]

    metatrain_task_distribution = TaskDistribution(tasks=metatrain_tasks_list,
                                                   task_probabilities=[1.0],
                                                   batch_size=meta_batch_size,
                                                   sample_with_replacement=True,
                                                   use_classes_only_once=True)

    metatest_task_distribution = TaskDistribution(tasks=metatest_tasks_list,
                                                  task_probabilities=[1.0],
                                                  batch_size=meta_batch_size,
                                                  sample_with_replacement=True,
                                                   use_classes_only_once=True)

    # TODO: split into validation and test!
    return metatrain_task_distribution, metatest_task_distribution, metatest_task_distribution
