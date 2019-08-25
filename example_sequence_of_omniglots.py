# NOTE: the code is slightly different when RL environments are used as tasks, as there is no more difference between
# train and test datasets, and because the agents need to interact with the environment directly.

from pyMeta.tasks.omniglot_tasks import create_omniglot_allcharacters_task_distribution
from pyMeta.tasks.miniimagenet_tasks import create_miniimagenet_task_distribution
from pyMeta.models.seq_fomaml import SeqFOMAMLMetaLearner
from pyMeta.networks import make_omniglot_cnn_model, make_miniimagenet_cnn_model
from pyMeta.core.task import TaskAsSequenceOfTasks
from pyMeta.core.task_distribution import TaskAsSequenceOfTasksDistribution

import sys, os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Force the batchnormalization layers to use statistics from the current minibatch only, instead of learnt accumulated
# statistics.
tf.keras.backend.set_learning_phase(1)

out_files_prefix = "seq10_"


metamodel = "FOMAML"  # FOMAML or Reptile

# When setting use_distributed=True, the program can be run with MPI as "mpirun -np NWORKERS python3 code.py". The effective batch size is NWORKERS*meta_batch_size;   add --oversubscribe or --use-hwthread-cpus to run more processes than physical cores (e.g., for virtual cores)
use_distributed = True

seed = 100


save_every_k_iterations = 1000
test_every_k_iterations = 100
model_save_filaname = "saved/"+out_files_prefix+"model.h5"


# Create a meta-dataset (sets of meta-train and meta-test tasks, for example classification tasks)

problem = "miniimagenet"  # omniglot, miniimagenet


if problem == "omniglot":
    # Sequence of Omniglots
    # We use the example of 5-shot, 5-way classification on Omniglot, as in the original Reptile paper.
    num_outer_metatraining_iterations = 10000
    outer_step_size = 0.001 if metamodel == 'FOMAML' else 1.0
    meta_batch_size = 5

    num_inner_training_iterations = 10  # 5
    inner_batch_size = 10
    inner_learning_rate = 0.001

    min_seq_length = 10
    max_seq_length = min_seq_length

    num_output_classes = 5

    # Individual tasks that are used to compose tasks-as-sequences.
    metatrain_task_distribution, metaval_task_distribution, metatest_tasks_distribution = \
                        create_omniglot_allcharacters_task_distribution('datasets/omniglot/omniglot.pkl',
                                                                        num_training_samples_per_class=5,
                                                                        num_test_samples_per_class=-1,
                                                                        num_training_classes=num_output_classes,
                                                                        batch_size=meta_batch_size)

elif problem == "miniimagenet":
    # Sequence of MiniImageNets
    # We use the example of 5-shot, 5-way classification on MiniImageNet
    num_outer_metatraining_iterations = 10000
    outer_step_size = 0.001 if metamodel == 'FOMAML' else 1.0
    meta_batch_size = 5

    min_seq_length = 10
    max_seq_length = min_seq_length

    num_output_classes = 5

    num_training_samples_per_class = 5
    num_test_samples_per_class = 15

    num_inner_training_iterations =  10
    inner_batch_size = num_training_samples_per_class * num_output_classes #10
    inner_learning_rate = 0.001 # 0.001 with Adam

    num_validation_batches = 5

    metatrain_task_distribution, metaval_task_distribution, metatest_tasks_distribution = \
                        create_miniimagenet_task_distribution('datasets/miniimagenet/miniimagenet.pkl',
                        num_training_samples_per_class=num_training_samples_per_class,
                        num_test_samples_per_class=num_test_samples_per_class,
                        num_training_classes=num_output_classes,
                        batch_size=meta_batch_size)


# Actual tasks that we will be using for meta-training: a sequence of tasks is considered a single task.
# The prototype objects are deepcopied by TaskAsSequenceOfTasksDistribution
train_task_prototype = TaskAsSequenceOfTasks(tasks_distribution=metatrain_task_distribution,
                                             min_length=min_seq_length,
                                             max_length=max_seq_length)
test_task_prototype = TaskAsSequenceOfTasks(tasks_distribution=metaval_task_distribution,
                                            min_length=min_seq_length,
                                            max_length=max_seq_length)

# Sample meta-batches of sequences of tasks
train_metabatch_distribution = TaskAsSequenceOfTasksDistribution(task_sequence_object=train_task_prototype,
                                                                 batch_size=meta_batch_size)
test_metabatch_distribution = TaskAsSequenceOfTasksDistribution(task_sequence_object=test_task_prototype,
                                                                batch_size=meta_batch_size)


# Omniglot model
if problem == "omniglot":
    model = make_omniglot_cnn_model(num_output_classes)

    optim = tf.keras.optimizers.Adam(lr=inner_learning_rate, beta_1=0.0)
    model.compile(optimizer=optim,
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['sparse_categorical_accuracy'])

elif problem == "miniimagenet":
    model = make_miniimagenet_cnn_model(num_output_classes)

    optim = tf.keras.optimizers.Adam(lr=inner_learning_rate, beta_1=0.0)
    model.compile(optimizer=optim,
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['sparse_categorical_accuracy'])



if metamodel == 'Reptile':
    # Reptile
    metalearner = ReptileMetaLearner(model=model,
                                     meta_learning_rate=outer_step_size,
                                     name="ReptileMetaLearner")

    if use_distributed:
        from pyMeta.models.distributed import DistributedReptileMetaLearner
        metalearner = DistributedReptileMetaLearner(model=model,
                                         meta_learning_rate=outer_step_size,
                                         name="DistributedReptileMetaLearner")

elif metamodel == 'FOMAML':
    # FOMAML
    optimizer = tf.train.AdamOptimizer(learning_rate=outer_step_size)
    metalearner = SeqFOMAMLMetaLearner(model=model,
                                       optimizer=optimizer,
                                       name="SeqFOMAMLMetaLearner")
    if use_distributed:
        from pyMeta.models.distributed import DistributedSeqFOMAMLMetaLearner
        metalearner = DistributedSeqFOMAMLMetaLearner(model=model,
                                                      optimizer=optimizer,
                                                      name="DistributedSeqFOMAMLMetaLearner")



if use_distributed:
    # Mute all but the first worker
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    #if rank > 0:
    #    sys.stdout = open(os.devnull, 'w')
    np.random.seed(seed + rank)
    tf.random.set_random_seed(seed + rank)




config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

sess.run(tf.global_variables_initializer())

# Load saved model
# model = tf.keras.models.load_model(model_save_filaname)


metalearner.initialize(session=sess)


# Main meta-training loop: for each outer iteration, we will sample a number of training tasks, then train on each of
# them (inner training loop) while recording their final test performance to track training. After all tasks in the
# meta-batch have been observed, the model is updated in the outer loop, and we proceed to the next outer iteration.
# Note that the focus is shifted on the outer training loop, with the inner one consisting of traditional
# single-task training.

all_results = []

last_time = time.time()
for outer_iter in range(num_outer_metatraining_iterations+1):
    # batch_tasks = omniglots_train_distribution.sample_batch(batch_size=meta_batch_size)
    batch_tasks = train_metabatch_distribution.sample_batch()

    # TODO: inefficient; we are solving each task sequentially, when we should rather use threads and parallelism!
    metabatch_results = []

    # avg_final_loss = np.asarray([0.0, 0.0])
    for task in batch_tasks:
        # Train on task for a number of num_inner_training_iterations iterations
        metalearner.task_begin(task)

        ret_info = task.fit_n_iterations(model, num_inner_training_iterations, inner_batch_size, return_weights_after_each_task=True)

        # if 'last_minibatch_loss' in ret_info:
        #     avg_final_loss += ret_info['last_minibatch_loss']

        metabatch_results.append(metalearner.task_end(task, results=ret_info, seq_fomaml_loss='loss_after_each_task'))

    # Update the meta-learner after all batch has been computed
    metalearner.update(metabatch_results)

    if outer_iter % test_every_k_iterations == 0:
        # Evaluate the meta-learner on a meta-validation set

        print("Time: ", time.time()-last_time)

        val_task_res = []
        # Evaluate over `num_validation_batches' validation meta-batches (=num_validation_batches*meta_batch_size random sequences)
        for val_task_num in range(num_validation_batches):
            batch_tasks = test_metabatch_distribution.sample_batch()

            for task in batch_tasks:
                metalearner.task_begin(task)

                task.fit_n_iterations(model, num_inner_training_iterations, inner_batch_size)
                out_dict = task.evaluate(model)
                val_task_res.append(out_dict)
        avg_dict = {k: np.mean([val_task_res[i][k]
                                for i in range(len(val_task_res))])
                    for k in val_task_res[0].keys()}
        all_results.append(avg_dict)

        num_tasks = min_seq_length
        xs = [x+1 for x in range(num_tasks)]
        plt.clf()
        fig, ax = plt.subplots()
        for i in range(0, len(all_results)-1, 1):
            color = plt.cm.Blues((float(i)+1)/len(all_results))
            ys = [all_results[i]['sparse_categorical_accuracy_'+str(xs[j]-1)] for j in range(num_tasks)]
            plt.plot(xs, ys, color=color)
        plt.plot(xs, [all_results[-1]['sparse_categorical_accuracy_'+str(xs[j]-1)] for j in range(num_tasks)], 'k--')
        plt.xticks(list(range(1, num_tasks+1)))
        plt.axis([min(xs), max(xs), 1.0/num_output_classes, 1.0])
        plt.xlabel("Task i")
        plt.ylabel("Accuracy")
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=True,
                       bottom=True, top=False, left=True, right=False)
        plt.savefig(out_files_prefix+"out.svg")
        plt.close()

        print('Iter: ', outer_iter, avg_dict)
        # '\n\tavg final loss across test tasks: ', np.mean(test_task_loss),
        # '\n\taverage test accuracy on test tasks: ',np.mean(test_task_accuracy)*100.0,'%')
        last_time = time.time()

    if outer_iter % save_every_k_iterations == 0:
        if (not use_distributed) or (use_distributed and MPI.COMM_WORLD.rank == 0):
            metalearner.task_begin(batch_tasks[0])  # copy back the initial parameters to the model's weights
            model.save(model_save_filaname)
