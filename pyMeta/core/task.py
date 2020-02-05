"""
Specification of the base Task interfaces.

+ Task: base Task class
+ ClassificationTask: generic class with utilities to wrap datasets for classification problems.
+ RLTask: generic class with utilities to wrap reinforcement learning environments.

+ TaskAsTaskSequence: wrapper class that builds "tasks" that represent a sequence of tasks. Each sub-task can be queried
  independently, to allow for testing their final performance after all tasks have been learnt sequentially. This class
  is designed to be a starting point for continual learning research.
"""

import sys

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




@tf.function
def train_on_batch(model, batch_X, batch_y):
    # TODO: note: because of tf.function, it's likely that model.losses will retain the very first value
    # it had before the first call to this function, and subsequent modifications may not be tracked.
    # This shouldn't be a problem, as this function is used only after model definition, compiling and wrapping
    # by meta-learner objects, but it's something to keep in mind.
    with tf.GradientTape() as tape:
        aux_loss = 0.0
        if len(model.losses) > 0:
            aux_loss = tf.add_n(model.losses)
        loss = tf.reduce_mean(model.loss(batch_y, model(batch_X, training=True))) + aux_loss
    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

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
            split_train_test : float [0,1], or <0
                On each reset, the instances in the dataset are first split into a train and test set. From those,
                num_training_samples_per_class and num_test_samples_per_class are sampled.
                If `split_train_test' < 0, then the split is automatically set to #train_samples / (#train_samples + #test_samples)

            Note: HACKY: a field 'label_offset' is always defined. It can be overridden after the object has been created. The purpose is to add a fixed starting offset for all labels, especially for use in multi_headed setups.
        """
        self.X = X
        self.y = y

        self._deepcopy_avoid_copying = ['X', 'y']

        self.num_training_classes = num_training_classes
        if self.num_training_classes >= len(set(self.y)):

            self.num_training_classes = -1
            print("WARNING: more training classes than available in the dataset were requested. \
                   All the available classes (", len(self.y), ") will be used.")

        self.split_train_test = split_train_test
        if self.split_train_test < 0:
            self.split_train_test = num_training_samples_per_class / (num_training_samples_per_class + num_test_samples_per_class)

        num_classes = self.num_training_classes if self.num_training_classes>0 else len(set(self.y))
        self.num_training_samples_per_class = num_training_samples_per_class
        if self.num_training_samples_per_class*num_classes >= len(self.y)*self.split_train_test:
            self.num_training_samples_per_class = -1
            print("WARNING: more training samples per class than available training instances were requested. \
                   All the available instances (", int(len(self.y)*self.split_train_test), ") will be used.")

        self.num_test_samples_per_class = num_test_samples_per_class
        if self.num_test_samples_per_class*num_classes >= len(self.y)*(1-self.split_train_test):
            self.num_test_samples_per_class = -1
            print("WARNING: more test samples per class than available test instances were requested. \
                   All the available instances (", int(len(self.y)*(1-self.split_train_test)), ") will be used.")

        self.label_offset = 0

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
            if hasattr(self,'_deepcopy_avoid_copying') and k not in self._deepcopy_avoid_copying:
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

        This is a utility method, but in general users should define their own function, as they may have
        different needs that are not met here (e.g., using regularizers)
        """

        for iteration in range(num_iterations):
            batch_X, batch_y = self.sample_batch(batch_size)

            # TFv1 : the following does not work in TFv2 if the model is defined with tf.function (e.g., w/ Keras'
            # functional API)
            ## loss = model.train_on_batch(tf.constant(batch_X), tf.constant(batch_y))

            # TFv2: optimized training step using @tf.function. The function needs to be defined outside this object
            # because task objects are created in large numbers, and it would require re-compiling the graph every
            # time.
            loss = train_on_batch(model, tf.constant(batch_X), tf.constant(batch_y))

            #print(model.metrics[0](batch_y, model(batch_X)))

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

        # Only valid with TFv1? (in this case, with the model defined using Keras function APIs
        #out = model.evaluate(test_X, test_y, batch_size=1000, verbose=0)
        #if not isinstance(out, list):
        #    out = [out]
        #out_dict = dict(zip(model.metrics_names, out))

        # TODO: wrap in a tf.function?
        # TODO: also: for... minibatch, run this code, which corresponds to .update_state
        out_dict = {}

        pred_test_y = model(test_X).numpy()

        if pred_test_y.shape[-1] > self.num_training_classes:
            # Likely multi-headed system. We will only let the outputs from output units in [offset, offset+num_training_classes) to pass, and zero-out the other outputs.
            pred_test_y[:, 0:self.label_offset] = 0.0
            pred_test_y[:, (self.label_offset+self.num_training_classes):] = 0.0

        if model.metrics_names[0] == 'loss':
            mets = [lambda true,pred: tf.reduce_mean(model.loss(true,pred)) + (tf.add_n(model.losses) if len(model.losses)>0 else 0)] + model.metrics
        for i in range(len(model.metrics_names)):
            if model.metrics_names[i] != 'loss':
                mets[i].reset_states()
            val = mets[i](test_y, pred_test_y)

            out_dict[model.metrics_names[i]] = val.numpy()

        return out_dict

    def sample_batch(self, batch_size):
        batch_indices = np.random.choice(self.train_indices, batch_size, replace=False)
        batch_X = self.X[batch_indices]
        batch_y = np.asarray([self.classes_ids.index(c)+self.label_offset
                              for c in self.y[batch_indices]], dtype=np.int64)
        return batch_X, batch_y

    def get_train_set(self):
        return self.X[self.train_indices], np.asarray([self.classes_ids.index(c)+self.label_offset
                                                       for c in self.y[self.train_indices]], dtype=np.int64)

    def get_test_set(self):
        return self.X[self.test_indices], np.asarray([self.classes_ids.index(c)+self.label_offset
                                                      for c in self.y[self.test_indices]], dtype=np.int64)


class ClassificationTaskFromFiles(ClassificationTask):
    def __init__(self, X, y, num_training_samples_per_class=-1, num_test_samples_per_class=-1,
                 num_training_classes=-1, split_train_test=0.8, input_parse_fn=None, num_parallel_processes=None):
        """
        input_parse_fn : function
            This function takes a filename as input and returns a loaded and processed sample.
        num_parallel_processes : int or None
            If the argument is not None, then minibatches are loaded from file and processed in a parallel Pool with
            'num_parallel_processes' processes.
        """
        self.input_parse_fn = input_parse_fn

        # NOTE: EXPERIMENTAL
        # TODO: debug: error too many open files, but my OS allows 1024, so something may be wrong.
        self.num_parallel_processes = num_parallel_processes

        super().__init__(X=X, y=y,
                         num_training_samples_per_class=num_training_samples_per_class,
                         num_test_samples_per_class=num_test_samples_per_class,
                         num_training_classes=num_training_classes, split_train_test=split_train_test)

    def reset(self):
        super().reset()

        if (self.num_parallel_processes is not None) and not hasattr(self, 'parallel_pool'):
            import multiprocessing
            self.parallel_pool = multiprocessing.Pool(self.num_parallel_processes)
            self._deepcopy_avoid_copying += ['parallel_pool']

    def _load_batch(self, indices):
        if self.num_parallel_processes is not None:
            # Load batch in Parallel
            filenames = [self.X[i] for i in indices]
            batch_x = self.parallel_pool.map(self.input_parse_fn, filenames)
        else:
            # Load batch in Sequence
            batch_x = [None]*len(indices)
            for i, ind in enumerate(indices):
                batch_x[i] = self.input_parse_fn(self.X[ind])
        batch_y = np.asarray([self.classes_ids.index(c)+self.label_offset for c in self.y[indices]], dtype=np.int64)
        return np.asarray(batch_x), batch_y

    def sample_batch(self, batch_size):
        batch_indices = np.random.choice(self.train_indices, batch_size, replace=False)
        batch_X, batch_y = self._load_batch(batch_indices)
        return batch_X, batch_y

    def get_train_set(self):
        batch_X, batch_y = self._load_batch(self.train_indices)
        return batch_X, batch_y

    def get_test_set(self):
        batch_X, batch_y = self._load_batch(self.test_indices)
        return batch_X, batch_y


class TaskAsSequenceOfTasks(Task):
    """
    The definition of a `Task' is ambiguous in the real world, where task boundaries are not always well defined,
    and when tasks often have a hierarchical structure.

    The `TaskAsSequenceOfTasks' class addresses this concern by representing a single `Task' as a sequence of tasks
    sampled from a distribution.
    """
    def __init__(self, tasks_distribution, min_length, max_length, multi_headed=False, num_classes_per_head=-1):
        """
        tasks_distribution : TaskDistribution
            Task distribution object to be used to sample tasks from when generating the sequences.
        min_length / max_length : int
            Minimum and maximum number of tasks in the sequence, inclusive.
        multi_headed : bool
            If multi_headed is True, all the labels of the tasks in the sequence will be added an offset of size num_classes_per_head for each task in the sequence. NOTE: the output layer of the model must be of size >= max_length*num_classes_per_head.
        num_classes_per_head : int
            Only valid if 'multi_headed' is True. See above for details.
        """
        self.tasks_distribution = tasks_distribution

        self.current_task_sequence = []

        self.min_length = min_length
        self.max_length = max_length

        self.multi_headed = multi_headed
        self.num_classes_per_head = num_classes_per_head

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

        if self.multi_headed:
            if len(set(self.current_task_sequence)) < len(self.current_task_sequence):
                print("ERROR: task objects must be independent for use with MULTI-HEADED TaskAsSequenceOfTasks")
                sys.exit()

            for i in range(len(self.current_task_sequence)):
                offset = i*self.num_classes_per_head
                self.current_task_sequence[i].label_offset = offset

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

        if hasattr(model, 'cl_reset'):
            model.cl_reset()

        ret_info = {}
        for task_i in range(len(self.current_task_sequence)):
            task = self.current_task_sequence[task_i]

            ret = task.fit_n_iterations(model, num_iterations, batch_size)

            for key in ret.keys():
                ret_info[key+"_"+str(task_i)] = ret[key]

            if return_weights_after_each_task:
                ret_info["weights_"+str(task_i)] = [v.numpy() for v in model.trainable_variables]

            # Callbacks for continual-learning algorithms
            if hasattr(model, 'run_after_task'):
                model.run_after_task()

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
