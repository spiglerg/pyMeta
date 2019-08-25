"""
Specification of the base Task interfaces.

+ Task: base Task class
+ ClassificationTask: generic class with utilities to wrap datasets for classification problems.
+ RLTask: generic class with utilities to wrap reinforcement learning environments.

+ TaskAsTaskSequence: wrapper class that builds "tasks" that represent a sequence of tasks. Each sub-task can be queried
  independently, to allow for testing their final performance after all tasks have been learnt sequentially. This class
  is designed to be a starting point for continual learning research.
"""

import numpy as np
import tensorflow as tf

from copy import deepcopy


class Task:
    def reset():
        """
        Reset a task after using it. For example, in case of ClassificationTask this would sample a new subset
        of instances for each class. In case of RLTask, it would force a reset the environment. Note that the
        environment is automatically reset on sampling new data during training.
        In case of derived classes (e.g., OmniglotTaskSampler or TaskAsTaskSequence), reset would re-sample a new task
        from the appropriate distribution.
        """
        pass


class ClassificationTask(Task):
    def __init__(self, X, y, num_training_samples_per_class=-1, num_test_samples_per_class=-1, num_training_classes=-1,
                 split_train_test=0.8):
        """
            X: ndarray [size, features, ...]
                Training dataset X.
            y: ndarray [size]
                Training dataset y.
            num_training_samples_per_class: int
                If -1, sample from the whole dataset. If >=1, the dataset will re-sample num_training_samples_per_class
                for each class at each reset, and only sample from them when queried, until the next reset.
                This is useful for, e.g., k-shot classification.
            num_test_samples_per_class: int
                If -1, sample from the whole dataset. If >=1, the dataset will re-sample num_test_samples_per_class
                for each class at each reset, and only sample from them when queried, until the next reset.
            num_training_classes: int
                If -1, use all the classes in `y'. If >=1, the dataset will re-sample num_training_classes at
                each reset, and only sample from them when queried, until the next reset.
            split_train_test : float [0,1]
                On each reset, the instances in the dataset are first split into a train and test set. From those,
                num_training_samples_per_class and num_test_samples_per_class are sampled.
        """
        self.X = X
        self.y = y

        self.num_training_samples_per_class = num_training_samples_per_class
        if self.num_training_samples_per_class >= len(self.y)*split_train_test:
            self.num_training_samples_per_class = -1
            print("WARNING: more training samples per class than available training instances were requested. \
                   All the available instances (", int(len(self.y)*split_train_test), ") will be used.")

        self.num_test_samples_per_class = num_test_samples_per_class
        if self.num_test_samples_per_class >= len(self.y)*(1-split_train_test):
            self.num_test_samples_per_class = -1
            print("WARNING: more test samples per class than available test instances were requested. \
                   All the available instances (", int(len(self.y)*(1-split_train_test)), ") will be used.")

        self.num_training_classes = num_training_classes
        if self.num_training_classes >= len(set(self.y)):
            self.num_training_classes = -1
            print("WARNING: more training classes than available in the dataset were requested. \
                   All the available classes (", len(self.y), ") will be used.")

        self.split_train_test = split_train_test

        self.reset()

    def __deepcopy__(self, memo):
        """
        We override __deepcopy__ to prevent deep-copying the dataset (self.X and self.y), which will thus be shared
        between all deepcopied instances of the object.
        This is best for performance, as the dataset is not modifying by the Task instances, which rather only store
        the per-reset indices of the train and test samples.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != 'X' and k != 'y':
                setattr(result, k, deepcopy(v, memo))
        setattr(result, 'X', self.X)
        setattr(result, 'y', self.y)
        return result

    def reset(self):
        classes_to_use = list(set(self.y))
        if self.num_training_classes >= 1:
            classes_to_use = np.random.choice(classes_to_use, self.num_training_classes, replace=False)

        self.train_indices = []
        self.test_indices = []

        # For each class, take list of indices, sample k
        for c in classes_to_use:
            class_indices = np.where(self.y == c)[0]
            np.random.shuffle(class_indices)

            train_test_separation = int(len(class_indices)*self.split_train_test)
            all_train_indices = class_indices[0:train_test_separation]
            all_test_indices = class_indices[train_test_separation:]

            if self.num_training_samples_per_class >= 1:
                all_train_indices = np.random.choice(all_train_indices,
                                                     self.num_training_samples_per_class,
                                                     replace=False)
            if self.num_test_samples_per_class >= 1:
                all_test_indices = np.random.choice(all_test_indices,
                                                    self.num_test_samples_per_class,
                                                    replace=False)

            self.train_indices.extend(all_train_indices)
            self.test_indices.extend(all_test_indices)

        np.random.shuffle(self.train_indices)
        np.random.shuffle(self.test_indices)

        # rename train and test indices so that they run in [0, num_of_classes)
        self.classes_ids = list(classes_to_use)

    def fit_n_iterations(self, model, num_iterations, batch_size):
        """
        Fir a model on data sampled from the current Task, performing `num_iterations' iterations of optimization
        with a minibatch of size `batch_size'.

        `model' must be a Keras model. It is a bit of a limitation, but it simplifies handling of the code
        quite significantly, so we opted for this solution, for the moment.
        """
        for iteration in range(num_iterations):
            batch_X, batch_y = self.sample_batch(batch_size)
            loss = model.train_on_batch(batch_X, batch_y)

        # Return the value of the loss function at the last minibatch observed.
        # TODO: perhaps it may be useful to also return the sum/mean loss over the training iterations.
        ret_info = {'last_minibatch_loss': loss}
        return ret_info

    def evaluate(self, model, evaluate_on_train=False):
        """
        Evaluate a Keras `model' on the current Task, according to the metrices provided when building the model.
        """
        if evaluate_on_train:
            test_X, test_y = self.get_train_set()
        else:
            test_X, test_y = self.get_test_set()
        out = model.evaluate(test_X, test_y, batch_size=1000, verbose=0)

        if not isinstance(out, list):
            out = [out]
        out_dict = dict(zip(model.metrics_names, out))
        return out_dict

    def sample_batch(self, batch_size):
        batch_indices = np.random.choice(self.train_indices, batch_size, replace=False)
        batch_X = self.X[batch_indices]
        batch_y = np.asarray([self.classes_ids.index(c) for c in self.y[batch_indices]], dtype=np.int64)
        return batch_X, batch_y

    def get_train_set(self):
        return self.X[self.train_indices], np.asarray([self.classes_ids.index(c)
                                                       for c in self.y[self.train_indices]], dtype=np.int64)

    def get_test_set(self):
        return self.X[self.test_indices], np.asarray([self.classes_ids.index(c)
                                                      for c in self.y[self.test_indices]], dtype=np.int64)


class TaskAsSequenceOfTasks(Task):
    """
    The definition of a `Task' is ambiguous in the real world, where task boundaries are not always well defined,
    and when tasks often have a hierarchical structure.

    The `TaskAsSequenceOfTasks' class addresses this concern by representing a single `Task' as a sequence of tasks
    sampled from a distribution.
    """
    def __init__(self, tasks_distribution, min_length, max_length):
        """
        tasks_distribution: TaskDistribution
            Task distribution object to be used to sample tasks from when generating the sequences.
        min_length / max_length: int
            Minimum and maximum number of tasks in the sequence, inclusive.
        """
        self.tasks_distribution = tasks_distribution

        self.current_task_sequence = []

        self.min_length = min_length
        self.max_length = max_length
        self.reset()

    def set_length_of_sequence(self, min_length, max_length):
        self.min_length = min_length
        self.max_length = max_length
        self.reset()

    def get_task_by_index(self, index):
        assert (index >= -1 and index < len(self.current_task_sequence)), "INVALID TASK INDEX"
        return self.current_task_sequence[index]

    def get_sequence_length(self):
        return len(self.current_task_sequence)

    def reset(self):
        """
        Generate a new sequence of tasks.
        """
        new_length = np.random.randint(self.min_length, self.max_length+1)
        self.current_task_sequence = self.tasks_distribution.sample_batch(batch_size=new_length)

    def sample_batch(self, batch_size):
        print("NOT IMPLEMENTED. Fit and evaluate the task using `fit_n_iterations' and `evaluate'")
        pass

    def fit_n_iterations(self, model, num_iterations, batch_size, return_weights_after_each_task=False):
        """
        Contrary to single `Task's, num_iterations here refers to the number of iterations to be performed
        on *each* task, sequentially, rather than the total number of iterations. The total number of steps
        executed is thus num_iterations*len(self.current_task_sequence) (which is between min_length and max_length,
        inclusive).

        return_weights_after_each_task : bool
            If true, returns ret_info['weights_<i>'] for i=0...(num_tasks-1)
        """

        ret_info = {}
        for task_i in range(len(self.current_task_sequence)):
            task = self.current_task_sequence[task_i]

            ret = task.fit_n_iterations(model, num_iterations, batch_size)

            for key in ret.keys():
                ret_info[key+"_"+str(task_i)] = ret[key]

            if return_weights_after_each_task:
                ret_info["weights_"+str(task_i)] = tf.keras.backend.get_session().run(model.trainable_variables)

        return ret_info

    def evaluate(self, model, evaluate_last_task_only=False):
        """
        Evaluate the performance of the model on all tasks in the sequence, unless evaluate_last_task_only=True.
        """

        tasks_to_evaluate = self.current_task_sequence
        if evaluate_last_task_only:
            tasks_to_evaluate = [self.current_task_sequence[-1]]

        out_dict = {}
        for task_i in range(len(tasks_to_evaluate)):
            task = tasks_to_evaluate[task_i]
            ret = task.evaluate(model)

            if len(tasks_to_evaluate) == 1:
                out_dict = ret
            else:
                for key in ret.keys():
                    out_dict[key+"_"+str(task_i)] = ret[key]

        return out_dict

    def get_test_set(self, task_index=-1):
        assert task_index >= -1 and task_index < len(self.current_task_sequence), "Invalid task index"
        return self.current_task_sequence[task_index].get_test_set()


class RLTask(Task):
    # TODO:
    def __init__(self):
        pass
# TODO: in tasks/ folder: GymRLTask;  but also create some gym wrappers (compatible to openai ones!) to resize or crop
# inputs, and pad outputs, so that all tasks will have the same I/O shape for use in meta-learning using the same
# model!

# HINT: if working heavily with Task objects, it may be useful to check for the type of Task being handled:
#   isinstance(task, ClassificationTask) -> use that interface
# This shouldn't be a big concern in general, as Task-s are designed to be interchangeable, but it is necessary to
# discriminate between Classification/Regression tasks and Reinforcement Learning tasks, that require very different
# handling.
