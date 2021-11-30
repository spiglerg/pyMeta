from pyMeta.tasks.dataset_from_files_tasks import create_omniglot_from_files_task_distribution
from pyMeta.tasks.omniglot_tasks import create_omniglot_allcharacters_task_distribution
from pyMeta.tasks.cifar100_tasks import create_cifar100_task_distribution
from pyMeta.tasks.miniimagenet_tasks import create_miniimagenet_task_distribution, create_miniimagenet_from_files_task_distribution
from pyMeta.contrib_tasks.core50 import create_core50_from_npz_task_distribution
from pyMeta.tasks.sinusoid_tasks import create_sinusoid_task_distribution
from pyMeta.metalearners.reptile import ReptileMetaLearner
from pyMeta.metalearners.fomaml import FOMAMLMetaLearner
from pyMeta.metalearners.implicit_maml import iMAMLMetaLearner
from pyMeta.networks import make_omniglot_cnn_model, make_miniimagenet_cnn_model, make_sinusoid_model, make_core50_cnn_model

import numpy as np
import tensorflow as tf

from absl import app, flags

# Force the batchnormalization layers to use statistics from the current minibatch only, instead of learnt accumulated
# statistics.
tf.keras.backend.set_learning_phase(1)

# Tensorflow 2.0 GPU memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)



FLAGS = flags.FLAGS

# Dataset and model options
flags.DEFINE_string('dataset', 'omniglot', 'omniglot or miniimagenet or sinusoid or cifar100 or core50')
flags.DEFINE_string('metamodel', 'fomaml', 'fomaml or reptile or imaml')

flags.DEFINE_integer('num_output_classes', 5, 'Number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_integer('num_train_samples_per_class', 5, 'Number of samples per class used in classification (e.g. 5-shot classification).')
flags.DEFINE_integer('num_test_samples_per_class', 15, 'Number of samples per class used in testing (e.g., evaluating a model trained on k-shots, on a different set of samples).')

flags.DEFINE_integer('num_evaluations', 100, 'Number of tasks to sample for the meta-test evaluation..')

# implicit-MAML (iMAML) specific options
flags.DEFINE_float('imaml_lambda_reg', 2.0, 'Value of lambda for the inner-loop L2 regularizer wrt to the initial parameters. Only used by iMAML. Original values are 2.0 for Omniglot and 0.5 for MiniImageNet.')
flags.DEFINE_integer('imaml_cg_steps', 5, 'Number of steps to run the iMAML optimizer for, in order to estimate the per-task meta-gradient. E.g., this usually refers to the number of iterations of Conjugate Gradient.')

# Inner-training options
flags.DEFINE_integer('num_inner_training_iterations', 5, 'Number of gradient descent steps to perform for each task in a meta-batch (inner steps).')
flags.DEFINE_integer('inner_batch_size', -1, 'Batch size: number of task-specific points sampled at each inner iteration. If <0, then it defaults to num_train_samples_per_class*num_output_classes.')
flags.DEFINE_float('inner_lr', 0.001, 'Learning rate of the inner optimizer. Default 0.01 for FOMAML, 1.0 for Reptile')

# Logging, saving, and testing options
flags.DEFINE_string('model_load_filename', 'saved/model', 'Path + filename where to save the model to.')

flags.DEFINE_integer('seed', '100', 'random seed.')



def main(argv):
    if FLAGS.inner_batch_size < 0:
        FLAGS.inner_batch_size = FLAGS.num_train_samples_per_class * FLAGS.num_output_classes
    FLAGS.dataset.lower()
    FLAGS.metamodel.lower()

    np.random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)

    meta_batch_size = FLAGS.num_evaluations

    meta_lr = 0.0

    def custom_sparse_categorical_cross_entropy_loss(y_true, y_pred):
        ## Implementation of sparse_categorial_cross_entropy_loss based on categorical_crossentropy,
        ## to work-around the limitation of the former when computing 2nd order derivatives (in the current
        ## Tensorflow implementation)
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), FLAGS.num_output_classes)
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred)


    # Create the dataset and network model
    if FLAGS.dataset == "omniglot":
        metatrain_task_distribution, metaval_task_distribution, metatest_task_distribution = \
                            create_omniglot_allcharacters_task_distribution(
                                                            'datasets/omniglot/omniglot.pkl',
                                                            num_training_samples_per_class=FLAGS.num_train_samples_per_class,
                                                            num_test_samples_per_class=FLAGS.num_test_samples_per_class,
                                                            num_training_classes=FLAGS.num_output_classes,
                                                            meta_batch_size=meta_batch_size)

        model = make_omniglot_cnn_model(FLAGS.num_output_classes)

        optim = tf.keras.optimizers.SGD(lr=FLAGS.inner_lr)
        if FLAGS.metamodel == "reptile":
            optim = tf.keras.optimizers.Adam(lr=FLAGS.inner_lr, beta_1=0.0)
        loss_function = custom_sparse_categorical_cross_entropy_loss
        metrics = ['sparse_categorical_accuracy']

    elif FLAGS.dataset == "cifar100":
        metatrain_task_distribution, metaval_task_distribution, metatest_task_distribution = \
                            create_cifar100_task_distribution(
                                                          num_training_samples_per_class=FLAGS.num_train_samples_per_class,
                                                          num_test_samples_per_class=FLAGS.num_test_samples_per_class,
                                                          num_training_classes=FLAGS.num_output_classes,
                                                          meta_train_test_split=0.7,
                                                          meta_batch_size=meta_batch_size)

        model = make_omniglot_cnn_model(FLAGS.num_output_classes)

        optim = tf.compat.v1.keras.optimizers.SGD(lr=FLAGS.inner_lr)
        if FLAGS.metamodel == "reptile":
            optim = tf.keras.optimizers.Adam(lr=FLAGS.inner_lr, beta_1=0.0)
        loss_function = custom_sparse_categorical_cross_entropy_loss  # tf.keras.losses.sparse_categorical_crossentropy
        metrics = ['sparse_categorical_accuracy']

    elif FLAGS.dataset == "miniimagenet":
        metatrain_task_distribution, metaval_task_distribution, metatest_task_distribution = \
                            create_miniimagenet_task_distribution('datasets/miniimagenet/miniimagenet.pkl',
                            #create_miniimagenet_from_files_task_distribution('datasets/miniimagenet_from_files/',
                            num_training_samples_per_class=FLAGS.num_train_samples_per_class,
                            num_test_samples_per_class=FLAGS.num_test_samples_per_class,
                            num_training_classes=FLAGS.num_output_classes,
                            meta_batch_size=meta_batch_size)

        model = make_miniimagenet_cnn_model(FLAGS.num_output_classes)

        optim = tf.keras.optimizers.SGD(lr=FLAGS.inner_lr)
        if FLAGS.metamodel == "reptile":
            optim = tf.keras.optimizers.Adam(lr=FLAGS.inner_lr, beta_1=0.0)
        loss_function = custom_sparse_categorical_cross_entropy_loss
        metrics = ['sparse_categorical_accuracy']

    elif FLAGS.dataset == "core50":
        metatrain_task_distribution, metaval_task_distribution, metatest_task_distribution = \
                            create_core50_from_npz_task_distribution('datasets/core50/',
                            num_training_samples_per_class=FLAGS.num_train_samples_per_class,
                            num_test_samples_per_class=FLAGS.num_test_samples_per_class,
                            num_training_classes=FLAGS.num_output_classes,
                            meta_batch_size=meta_batch_size)

        model = make_core50_cnn_model(FLAGS.num_output_classes)
        model = make_miniimagenet_cnn_model(FLAGS.num_output_classes, input_shape=(128,128,3)) # this works well and it's fast, but it achieves lower performance than the other network (52% instead of 60%?)

        optim = tf.keras.optimizers.SGD(lr=FLAGS.inner_lr)
        loss_function = custom_sparse_categorical_cross_entropy_loss
        metrics = ['sparse_categorical_accuracy']

    elif FLAGS.dataset == "sinusoid":
        metatrain_task_distribution, metaval_task_distribution, metatest_task_distribution = \
                            create_sinusoid_task_distribution(
                                                              min_amplitude=0.1,
                                                              max_amplitude=5.0,
                                                              min_phase=0.0,
                                                              max_phase=2 * np.pi,
                                                              min_x=-5.0,
                                                              max_x=5.0,
                                                              num_training_samples=FLAGS.num_train_samples_per_class,
                                                              num_test_samples=FLAGS.num_test_samples_per_class,
                                                              num_test_tasks=100,
                                                              meta_batch_size=meta_batch_size)

        model = make_sinusoid_model()

        optim = tf.keras.optimizers.Adam(lr=FLAGS.inner_lr, beta_1=0.0)
        loss_function = tf.keras.losses.mean_squared_error
        metrics = []

    else:
        print("ERROR: training task not recognized [", FLAGS.dataset, "]")
        sys.exit()


    # Setup the meta-learner
    if FLAGS.metamodel == 'reptile':
        optimizer = tf.keras.optimizers.SGD(learning_rate=meta_lr)
        metalearner = ReptileMetaLearner(model=model,
                                         optimizer=optimizer,
                                         name="ReptileMetaLearner")

    elif FLAGS.metamodel == 'fomaml':
        optimizer = tf.keras.optimizers.Adam(learning_rate=meta_lr)  # , beta_1=0.0)
        metalearner = FOMAMLMetaLearner(model=model,
                                        optimizer=optimizer,
                                        name="FOMAMLMetaLearner")
    elif FLAGS.metamodel == 'imaml':
        optimizer = tf.keras.optimizers.Adam(learning_rate=meta_lr)  # , beta_1=0.0)
        #optimizer = tf.keras.optimizers.SGD(learning_rate=meta_lr)
        metalearner = iMAMLMetaLearner(model=model,
                                      optimizer=optimizer,
                                      lambda_reg = FLAGS.imaml_lambda_reg, #0.5, #2.0,
                                      n_iters_optimizer = FLAGS.imaml_cg_steps,
                                      name="iMAMLMetaLearner")


    # The model should be compiled AFTER being wrapped by a meta-learner, as the meta-learner may add special ops
    # or regularizers to the model.
    model.compile(optimizer=optim,
                  loss=loss_function,
                  metrics=metrics)

    model.summary()
    print("Meta model: ", FLAGS.metamodel)
    print("Problem: ", FLAGS.dataset)



    model.load_weights(FLAGS.model_load_filename)

    metalearner.initialize()



    # Evaluate the meta-learner on a set of the meta-test set
    test_task_loss = []
    test_task_accuracy = []

    batch_validation = metatest_task_distribution.sample_batch()

    for task in batch_validation:
        metalearner.task_begin(task)

        task.fit_n_iterations(model, tf.constant(FLAGS.num_inner_training_iterations), tf.constant(FLAGS.inner_batch_size))
        out_dict = task.evaluate(model)

        test_task_loss.append(out_dict['loss'])
        if 'sparse_categorical_accuracy' in out_dict:
            test_task_accuracy.append(out_dict['sparse_categorical_accuracy'])

    print('avg final loss across test tasks: ', np.mean(test_task_loss),
          '\naverage test accuracy on test tasks: ', np.mean(test_task_accuracy)*100.0, '%')



if __name__ == '__main__':
    app.run(main)
