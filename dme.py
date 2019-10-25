# NOTE: the code is slightly different when RL environments are used as tasks, as there is no more difference between
# train and test datasets, and because the agents need to interact with the environment directly.

from pyMeta.tasks.dataset_from_files_tasks import create_omniglot_from_files_task_distribution
from pyMeta.tasks.omniglot_tasks import create_omniglot_allcharacters_task_distribution
from pyMeta.tasks.cifar100_tasks import create_cifar100_task_distribution
from pyMeta.tasks.miniimagenet_tasks import create_miniimagenet_task_distribution
from pyMeta.tasks.sinusoid_tasks import create_sinusoid_task_distribution
from pyMeta.metalearners.reptile import ReptileMetaLearner
from pyMeta.metalearners.fomaml import FOMAMLMetaLearner
from pyMeta.metalearners.implicit_maml import iMAMLMetaLearner
from pyMeta.core.task import ClassificationTask
from pyMeta.core.task_distribution import TaskDistribution
from pyMeta.networks import make_omniglot_cnn_model, make_miniimagenet_cnn_model, make_sinusoid_model

import sys, os
import time
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import flags

# Force the batchnormalization layers to use statistics from the current minibatch only, instead of learnt accumulated
# statistics.
tf.keras.backend.set_learning_phase(1)


FLAGS = flags.FLAGS

# Dataset and model options
flags.DEFINE_string('metamodel', 'imaml', 'fomaml or reptile or imaml')

flags.DEFINE_integer('num_train_samples_per_class', 10, 'number of samples per class used in classification (e.g. 5-shot classification).')
flags.DEFINE_integer('num_test_samples_per_class', 10, 'number of samples per class used in testing (e.g., evaluating a model trained on k-shots, on a different set of samples).')

# Meta-training options
flags.DEFINE_integer('num_outer_metatraining_iterations', 100, 'number of iterations in the outer (meta-training) loop.')
flags.DEFINE_integer('meta_batch_size', 1, 'meta-batch size: number of tasks sampled at each meta-iteration.')
flags.DEFINE_float('meta_lr', 0.001, 'learning rate of the meta-optimizer ("outer" step size). Default 0.001 for FOMAML, 1.0 for Reptile') # 0.1 for omniglot


# Inner-training options
flags.DEFINE_integer('num_inner_training_iterations', 10, 'number of gradient descent steps to perform for each task in a meta-batch (inner steps).')
flags.DEFINE_integer('inner_batch_size', -1, 'batch size: number of task-specific points sampled at each inner iteration. If <0, then it defaults to num_train_samples_per_class*num_output_classes.')
flags.DEFINE_float('inner_lr', 0.001, 'learning rate of the inner optimizer. Default 0.01 for FOMAML, 1.0 for Reptile')

flags.DEFINE_integer('seed', '100', 'random seed.')


if FLAGS.inner_batch_size < 0:
    FLAGS.inner_batch_size = 8

FLAGS.metamodel.lower()

np.random.seed(FLAGS.seed)
tf.random.set_random_seed(FLAGS.seed)





class EasyTask(ClassificationTask):
    def __init__(self):
        """
        The test set is generated evenly from min_x to max_x.
        """
        self.train_indices = np.arange(-10, 10+1, 20/32) 
        self.reset()

    def reset(self):
        # Generate a new sinusoid training set
        self.a = np.random.random()*2-1
        self.b = np.random.random()*2-1

    def sample_batch(self, batch_size):
        x = np.random.randint(-10, 10+1, size=(batch_size,))
        y = self.a*x + self.b
        x = np.expand_dims(x, -1)
        y = np.expand_dims(y, -1)
        return x, y

    def get_train_set(self):
        x = np.arange(-10, 10+1, 20/32) 
        y = self.a*x + self.b
        x = np.expand_dims(x, -1)
        y = np.expand_dims(y, -1)
        return x, y

    def get_test_set(self):
        return self.get_train_set()




metatrain_tasks_list = [EasyTask()] # defaults to num_train / (num_train+num_test)
metatrain_task_distribution = TaskDistribution(tasks=metatrain_tasks_list,
                                               task_probabilities=[1.0],
                                               batch_size=FLAGS.meta_batch_size,
                                               sample_with_replacement=True)
metatest_task_distribution = metatrain_task_distribution
metaval_task_distribution = metatrain_task_distribution




model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=[1]),
    tf.keras.layers.Dense(1, activation='linear')
])




optim = tf.keras.optimizers.SGD(lr=FLAGS.inner_lr)
loss_function = tf.keras.losses.mean_squared_error
metrics = []



optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.meta_lr)  # , beta1=0.0)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.meta_lr)
metalearner = iMAMLMetaLearner(model=model,
                              optimizer=optimizer,
                              lambda_reg = 1.0,
                              name="iMAMLMetaLearner")

# The model should be compiled AFTER being wrapped by a meta-learner, as the meta-learner may add special ops
# or regularizers to the model.
model.compile(optimizer=optim,
              loss=loss_function,
              metrics=['sparse_categorical_accuracy'])



# Tensorflow Session and initialization (all variables, and meta-learner's initial state)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

metalearner.initialize(session=sess)


model.summary()
print("Meta model: ", FLAGS.metamodel)



# Main meta-training loop: for each outer iteration, we will sample a number of training tasks, then train on each of
# them (inner training loop) while recording their final test performance to track training. After all tasks in the
# meta-batch have been observed, the model is updated in the outer loop, and we proceed to the next outer iteration.
# Note that the focus is shifted on the outer training loop, with the inner one consisting of traditional
# single-task training.
last_time = time.time()
for outer_iter in range(FLAGS.num_outer_metatraining_iterations+1):
    meta_batch = metatrain_task_distribution.sample_batch()

    # META-TRAINING over batch
    # TODO: inefficient; we are solving each task sequentially, when we should rather do it in parallel
    # However it may be better to do it this way for few-shot classification problems, where few inner iterations are
    # used.
    metabatch_results = []
    avg_loss_lastbatch = np.asarray([0.0, 0.0])
    for task in meta_batch:
        # Train on task for a number of num_inner_training_iterations iterations
        metalearner.task_begin(task)

        ret_info = task.fit_n_iterations(model, FLAGS.num_inner_training_iterations, FLAGS.inner_batch_size)

        if 'last_minibatch_loss' in ret_info:
            avg_loss_lastbatch += ret_info['last_minibatch_loss']

        metabatch_results.append(metalearner.task_end(task))

    print(avg_loss_lastbatch)

    # Update the meta-learner after all batch has been computed
    metalearner.update(metabatch_results)




