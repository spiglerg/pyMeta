import sys
sys.path.insert(0, "pymeta_repository")

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.python.platform import flags

from pyMeta.core.task import TaskAsSequenceOfTasks
from pyMeta.core.task_distribution import TaskDistribution, TaskAsSequenceOfTasksDistribution
from pyMeta.tasks.omniglot_tasks import create_omniglot_allcharacters_task_distribution
from pyMeta.tasks.miniimagenet_tasks import create_miniimagenet_task_distribution
from pyMeta.models.reptile import ReptileMetaLearner
from pyMeta.models.fomaml import FOMAMLMetaLearner
from pyMeta.models.seq_fomaml import SeqFOMAMLMetaLearner
from pyMeta.networks import make_omniglot_cnn_model, make_miniimagenet_cnn_model

# Force the batchnormalization layers to use statistics from the current minibatch only, instead of learnt accumulated
# statistics.
tf.keras.backend.set_learning_phase(1)


FLAGS = flags.FLAGS

# Dataset and model options
flags.DEFINE_string('dataset', 'omniglot', 'omniglot or miniimagenet')

flags.DEFINE_integer('num_output_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_integer('num_train_samples_per_class', 5, 'number of samples per class used in classification (e.g. 5-shot classification).')
flags.DEFINE_integer('num_test_samples_per_class', 15, 'number of samples per class used in testing (e.g., evaluating a model trained on k-shots, on a different set of samples).')

flags.DEFINE_string('seq_fomaml_loss', 'loss_after_each_task', 'only valid for seq_fomaml. Can be "plain" or "loss_after_each_task" or "sampled_loss_after_each_task" or "all_losses_after_each_task". Default is "plain", which is the most direct adaptation of FOMAML to sequences of tasks. "loss_after_each_task" is the suggested one.')

# Sequence of tasks options
flags.DEFINE_integer('min_seq_length', 10, 'the length of the task sequences is sampled uniformly between [min_seq_length, max_seq_length], inclusive. If max_seq_length<0, then it is taken max_seq_length=min_seq_length.')
flags.DEFINE_integer('max_seq_length', -1, 'the length of the task sequences is sampled uniformly between [min_seq_length, max_seq_length], inclusive. If max_seq_length<0, then it is taken max_seq_length=min_seq_length.')

# Meta-training options
flags.DEFINE_integer('num_outer_metatraining_iterations', 20000, 'number of iterations in the outer (meta-training) loop.')
flags.DEFINE_integer('meta_batch_size', 5, 'meta-batch size: number of tasks sampled at each meta-iteration.')
flags.DEFINE_float('meta_lr', 0.001, 'learning rate of the meta-optimizer. Default 0.001 for FOMAML, 1.0 for Reptile')

flags.DEFINE_integer('num_validation_batches', 10, 'number of batches to sample from, and average over, when validating the performance of the model at regular intervals.')

# Inner-training options
flags.DEFINE_integer('num_inner_training_iterations', 10, 'number of gradient descent steps to perform for each task in a meta-batch (inner steps).')
flags.DEFINE_integer('inner_batch_size', 10, 'batch size: number of task-specific points sampled at each inner iteration.')
flags.DEFINE_float('inner_lr', 0.001, 'learning rate of the inner optimizer. Default 0.001 for FOMAML, 1.0 for Reptile')

# Logging, saving, and testing options
flags.DEFINE_integer('save_every_k_iterations', 1000, 'the model is saved every k iterations.')
flags.DEFINE_integer('test_every_k_iterations', 100, 'the performance of the model is evaluated every k iterations.')

flags.DEFINE_boolean('full_evaluation', True, 'if True, performs a more comprehensive set of tests on validation iterations.')

flags.DEFINE_string('out_files_folder', 'saved/', 'folder to save output files to.')
flags.DEFINE_string('out_files_prefix', '', 'prefix string to use for output files.')

flags.DEFINE_integer('seed', '100', 'random seed.')


if FLAGS.inner_batch_size < 0:
    FLAGS.inner_batch_size = FLAGS.num_train_samples_per_class * FLAGS.num_output_classes
if FLAGS.max_seq_length < 0:
    FLAGS.max_seq_length = FLAGS.min_seq_length
FLAGS.dataset.lower()

np.random.seed(FLAGS.seed)
tf.random.set_random_seed(FLAGS.seed)



from pyMeta.contrib_tasks.permuted_mnist_tasks import create_permuted_mnist_task_distribution, make_permuted_mnist_mlp_model
metatrain_task_distribution, metaval_task_distribution, metatest_tasks_distribution = \
                    create_permuted_mnist_task_distribution(
                                                    num_training_samples_per_class=FLAGS.num_train_samples_per_class,
                                                    num_test_samples_per_class=FLAGS.num_test_samples_per_class,
                                                    num_training_classes=FLAGS.num_output_classes,
                                                    meta_batch_size=FLAGS.meta_batch_size)

model = make_permuted_mnist_mlp_model(FLAGS.num_output_classes)
optim = tf.keras.optimizers.SGD(lr=FLAGS.inner_lr)
#optim = tf.train.experimental.enable_mixed_precision_graph_rewrite(optim)
model.compile(optimizer=optim,
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['sparse_categorical_accuracy'])



# Create the "sequence" Tasks, that is tasks that stand for whole sequences of other tasks
# A sequence of tasks is considered a single task
# The prototype objects are deepcopied by TaskAsSequenceOfTasksDistribution
train_task_prototype = TaskAsSequenceOfTasks(tasks_distribution=metatrain_task_distribution,
                                             min_length=FLAGS.min_seq_length,
                                             max_length=FLAGS.max_seq_length)
val_task_prototype = TaskAsSequenceOfTasks(tasks_distribution=metaval_task_distribution,
                                           min_length=FLAGS.min_seq_length,
                                           max_length=FLAGS.max_seq_length)


# Create a distribution over sequences, so that a batch of sequences of tasks can be sampled
train_metabatch_distribution = TaskAsSequenceOfTasksDistribution(task_sequence_object=train_task_prototype,
                                                                 batch_size=FLAGS.meta_batch_size)
val_metabatch_distribution = TaskAsSequenceOfTasksDistribution(task_sequence_object=val_task_prototype,
                                                               batch_size=FLAGS.meta_batch_size)


# Create the meta-learner: Seq-FOMAML
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.meta_lr)
metalearner = SeqFOMAMLMetaLearner(model=model,
                                   optimizer=optimizer,
                                   name="SeqFOMAMLMetaLearner")


# Tensorflow Session and initialization (variables, and then meta-learner's initial state)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.log_device_placement = True
sess = tf.InteractiveSession(config=config)

sess.run(tf.global_variables_initializer())

# Load saved model
# model = tf.keras.models.load_model(os.path.join(FLAGS.out_files_folder, FLAGS.out_files_prefix+"model.h5"))

metalearner.initialize(session=sess)


model.summary()
print("Problem: ", FLAGS.dataset)


# Main meta-training loop: for each outer iteration, we will sample a number of training tasks, then train on each of
# them (inner training loop) while recording their final test performance to track training. After all tasks in the
# meta-batch have been observed, the model is updated in the outer loop, and we proceed to the next outer iteration.
# Note that the focus is shifted on the outer training loop, with the inner one consisting of traditional
# single-task training.
all_results = []
last_time = time.time()
for outer_iter in range(FLAGS.num_outer_metatraining_iterations+1):
    # batch_tasks = omniglots_train_distribution.sample_batch(batch_size=FLAGS.meta_batch_size)
    batch_tasks = train_metabatch_distribution.sample_batch()

    # META-TRAINING over batch
    # TODO: inefficient; we are solving each task sequentially, when we should rather use threads and parallelism!
    metabatch_results = []
    for task in batch_tasks:
        # Train on task for a number of FLAGS.num_inner_training_iterations iterations
        metalearner.task_begin(task)

        ret_info = task.fit_n_iterations(model, FLAGS.num_inner_training_iterations, FLAGS.inner_batch_size, return_weights_after_each_task=True)

        metabatch_results.append(metalearner.task_end(task, results=ret_info, seq_fomaml_loss=FLAGS.seq_fomaml_loss))

    # Update the meta-learner after all batch has been computed
    metalearner.update(metabatch_results)


    ## META-TESTING every `test_every_k_iterations' iterations
    if outer_iter % FLAGS.test_every_k_iterations == 0:
        # Evaluate the meta-learner on a set of `metatest_tasks_list'
        print("Time: ", time.time()-last_time)

        test_task_res = []
        # Evaluate over K test meta-batches (=K*FLAGS.meta_batch_size random sequences)
        # Compute the accuracy of each task at the end of the sequence.
        # NOTE: usually the performance on all previous tasks is evaluated after training on each new task in the
        # sequence; however, this is slow ( O(N^2) ). This evaluation can be enabled with the flag --full_evaluation=True
        # TODO: --full_evaluation;  currently ignored
        for test_task_num in range(FLAGS.num_validation_batches):
            batch_tasks = val_metabatch_distribution.sample_batch()

            for task in batch_tasks:
                metalearner.task_begin(task)

                task.fit_n_iterations(model, FLAGS.num_inner_training_iterations, FLAGS.inner_batch_size)
                out_dict = task.evaluate(model)
                test_task_res.append(out_dict)

        avg_dict = {k: np.mean([test_task_res[i][k]
                                for i in range(len(test_task_res))])
                    for k in test_task_res[0].keys()}
        all_results.append(avg_dict)

        num_tasks = FLAGS.min_seq_length
        if num_tasks > 1:
            xs = [x+1 for x in range(num_tasks)]
            plt.clf()
            fig, ax = plt.subplots()
            for i in range(0, len(all_results)-1, 1):
                color = plt.cm.Blues((float(i)+1)/len(all_results))
                ys = [all_results[i]['sparse_categorical_accuracy_'+str(xs[j]-1)] for j in range(num_tasks)]
                plt.plot(xs, ys, color=color)
            plt.plot(xs, [all_results[-1]['sparse_categorical_accuracy_'+str(xs[j]-1)] for j in range(num_tasks)], 'k--')
            plt.xticks(list(range(1, num_tasks+1)))
            plt.axis([min(xs), max(xs), 1.0/FLAGS.num_output_classes, 1.0])
            plt.xlabel("Task i")
            plt.ylabel("Accuracy")
            ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=True,
                           bottom=True, top=False, left=True, right=False)
            plt.savefig(FLAGS.out_files_prefix+"out.svg")
            plt.close()

        print('Iter: ', outer_iter, avg_dict)
        last_time = time.time()

    if outer_iter % FLAGS.save_every_k_iterations == 0:
        metalearner.task_begin(batch_tasks[0])  # copy back the initial parameters to the model's weights
        model.save(os.path.join(FLAGS.out_files_folder, FLAGS.out_files_prefix+"_model.h5"))
