"""
Author:         Dennis Broekhuizen, Tilburg University
Credits:        Giacomo Spigler, pyMeta: https://github.com/spiglerg/pyMeta
Description:    Load the CORe50 dataset into memory and create a distribution
                of ClassificationTasks.

Main function to use:
def create_core50_from_npz_task_distribution(path_to_dataset,
                                             batch_size=32,
                                             num_training_samples_per_class=10,
                                             num_test_samples_per_class=-1,
                                             num_training_classes=20,
                                             meta_batch_size=5)
which returns the 3 TaskDistributions (train, validation, and test).

For download instructions, please refer to the original repository: https://github.com/DennisBroekhuizen/pyMetaCORe50
1) Download the dataset (core50_imgs.npz, paths.pkl @ https://vlomonaco.github.io/core50/index.html#download)
2) Move the files into datasets/core50/
"""

import pickle as pkl
import os
import numpy as np
import tensorflow as tf
import cv2

from pyMeta.core.task_distribution import TaskDistribution
from pyMeta.core.task import ClassificationTaskFromFiles

core50_images = []

def load_npz_file(path_to_dataset):
    """
    Loads the CORe50 npz file into memory.
    Requirements:
      - Path to folder with CORe50 files: paths.pkl and core50_imgs.npz
    """
    # Load the paths file.
    pkl_file = open(path_to_dataset+'paths.pkl', 'rb')
    paths = pkl.load(pkl_file)

    # Load the image file.
    imgs = np.load(path_to_dataset+'core50_imgs.npz')['x']

    return imgs, paths


def process_npz_img(npz_index):
    return np.asarray(core50_images[npz_index], dtype=np.float32)


class ClassificationTaskCORe50(ClassificationTaskFromFiles):
    def __init__(self, X, y, num_training_samples_per_class=-1, num_test_samples_per_class=-1, num_training_classes=-1,
                 split_train_test=0.8, input_parse_fn=None, background_labels=None):

        self.background_labels = background_labels

        super().__init__(X=X, y=y,
                         num_training_samples_per_class=num_training_samples_per_class,
                         num_test_samples_per_class=num_test_samples_per_class,
                         num_training_classes=num_training_classes, split_train_test=split_train_test, input_parse_fn=input_parse_fn)

    def reset(self):
        # Substract all unique classes.
        classes_to_use = list(set(self.y))

        # Check if train classes >= 1 and choose classes to use.
        if self.num_training_classes >= 1:
            classes_to_use = np.random.choice(classes_to_use, self.num_training_classes, replace=False)

        # Pre-define backgrounds to use for the test set.
        test_bkg = [3, 7, 10]
        test_bkg_indices = []
        for i in range(len(test_bkg)):
            test_bkg_indices.extend(np.where(self.background_labels==test_bkg[i])[0])
        test_bkg_indices = np.asarray(test_bkg_indices)

        # Pre-define backgrounds to use for the train set.
        train_bkg = [1, 2, 4, 5, 6, 8, 9, 11]
        train_bkg_indices = []
        for i in range(len(train_bkg)):
            train_bkg_indices.extend(np.where(self.background_labels==train_bkg[i])[0])
        train_bkg_indices = np.asarray(train_bkg_indices)

        # Define lists to save train and test samples (indices).
        self.train_indices = []
        self.test_indices = []

        # For each class, take list of indices, sample k.
        for c in classes_to_use:
            # List of class indices equal to current class in loop.
            class_indices = np.where(self.y==c)[0]

            # Check and use the intersection between current class and background indices.
            all_train_indices = list(set(list(train_bkg_indices)).intersection(set(list(class_indices))))
            all_test_indices = list(set(list(test_bkg_indices)).intersection(set(list(class_indices))))

            # Randomly choose train and test samples of current class.
            if self.num_training_samples_per_class >= 1:
                all_train_indices = np.random.choice(all_train_indices,
                                                     self.num_training_samples_per_class,
                                                     replace=False)
            if self.num_test_samples_per_class >= 1:
                all_test_indices = np.random.choice(all_test_indices,
                                                    self.num_test_samples_per_class,
                                                    replace=False)

            # Add samples of class to list of all samples.
            self.train_indices.extend(all_train_indices)
            self.test_indices.extend(all_test_indices)

        # Randomly shuffle the final train and test sample lists.
        np.random.shuffle(self.train_indices)
        np.random.shuffle(self.test_indices)

        # Rename train and test indices so that they run in [0, num_of_classes).
        self.classes_ids = list(classes_to_use)


def create_core50_from_npz_task_distribution(path_to_dataset,
                                             batch_size=32,
                                             num_training_samples_per_class=10,
                                             num_test_samples_per_class=-1,
                                             num_training_classes=20,
                                             meta_batch_size=5):

    imgs, paths = load_npz_file(path_to_dataset)

    global core50_images
    core50_images = imgs

    def get_session_objects(session_num, path_file):
        session_indexes = []
        session_labels = []
        for index, path in enumerate(path_file):
            splitted_path = path.split('/')
            if splitted_path[0] == 's'+str(session_num):
                for i in range(1, 51):
                    if splitted_path[1] == 'o'+str(i):
                        session_indexes.append(index)
                        session_labels.append(i)
        return session_indexes, session_labels


    def dataset_from_npz(session_nums, path_file):
        # Object index numbers in npz file.
        X_indexes = []

        # Object labels.
        y = []

        # Background (session) labels.
        b = []

        for session_num in session_nums:
            session_indexes, session_labels = get_session_objects(session_num, path_file)
            X_indexes.extend(session_indexes)
            y.extend(session_labels)
            for i in range(len(session_indexes)):
                b.append(session_num)

        X_indexes = np.asarray(X_indexes, dtype=np.int32)
        y = np.asarray(y, dtype=np.int32)
        b = np.asarray(b, dtype=np.int32)

        return X_indexes, y, b


    # Pre-define backround sessions to use.
    all_sessions = []
    for i in range(1, 12):
        all_sessions.append(i)

    X_indexes, y, b = dataset_from_npz(session_nums=all_sessions, path_file=paths)

    # Split indexes: first 40 objects train set & last 10 objects for test set.
    train_indexes = np.where(y<=40)[0]
    test_indexes = np.where(y>40)[0]

    # Split the dataset.
    trainX = X_indexes[train_indexes]
    trainY = y[train_indexes]
    trainB = b[train_indexes]

    testX = X_indexes[test_indexes]
    testY = y[test_indexes]
    testB = b[test_indexes]


    # Create ClassificationTask objects
    metatrain_tasks_list = [ClassificationTaskCORe50(trainX,
                                               trainY,
                                               num_training_samples_per_class,
                                               num_test_samples_per_class,
                                               num_training_classes,
                                               split_train_test=-1,
                                               input_parse_fn=process_npz_img, # defaults to num_train / (num_train+num_test)
                                               background_labels=trainB)]
    metatest_tasks_list = [ClassificationTaskCORe50(testX,
                                              testY,
                                              num_training_samples_per_class,
                                              num_test_samples_per_class,
                                              num_training_classes,
                                              split_train_test=-1,
                                              input_parse_fn=process_npz_img,
                                              background_labels=testB)]

    # Create TaskDistribution objects that wrap the ClassificationTask objects to produce meta-batches of tasks
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
