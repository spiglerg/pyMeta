"""
Implementation of the iMAML (implicit-MAML) algorithm for meta-learning.
Rajeswaran*, Finn*, Kakade, and Levine (2019) - https://arxiv.org/pdf/1909.04630

WARNING: this code is super ugly and hacky, and can be simplified greatly.
WARNING2: l-BFGS was preferred to CG (contrary to the original iMAML paper) because (1) it was found
  to converge faster / work better (although that may depend on the implementations I tried), and (2)
  an implementation of l-BFGS is provided within tensorflow_probability, with a great interface with
  Tensorflow's computational graphs.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from pyMeta.core.meta_learner import GradBasedMetaLearner
from pyMeta.core.task import TaskAsSequenceOfTasks




class iMAMLMetaLearner(GradBasedMetaLearner):
    def __init__(self, model, optimizer=tf.train.AdamOptimizer(learning_rate=0.001), lambda_reg=1.0, n_iters_optimizer=5, name="FOMAMLMetaLearner"):
        """
        IMPORTANT: the Keras model passed to iMAML must be re-compiled after creating the metalearner object, as it
        adds a new loss (L2 regularization wrt to the initial parameters).

        In general, meta-learners objects should be created before tf.global_variables_initializer() is called, in
        case variables have to be created.

        This meta-learner should be used as follows:
        + Instantiate object (before calling tf.global_variables_initializer() )
        + Initialize object, after tf.global_variables_initializer()
            metalearner.initialize()
        + For each meta-learning iteration:
            + Go through each task in the meta-batch
            + Tell the meta-learner a new task is starting
                metalearner.task_begin(task)
            + Train the task
            + Tell the meta-learner the task has finished training, and collect the returned values
                results.append(metalearner.task_end(task))
        + Perform a meta-update
            metalearner.update(results)

        optimizer: tf.train.Optimizer objects
            The optimizer to use for meta-learning updates (outer loop).
        model: tf.keras.Model
            The model to meta-optimize.
        """
        self.model = model
        self.optimizer = optimizer

        self.lambda_reg = lambda_reg
        self.n_iters_optimizer = n_iters_optimizer

        self.target_placeholder = None

        # Update op to change the current initial parameters (weights initialization)
        self._placeholders = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape())
                              for v in self.model.trainable_variables]
        assigns = [tf.assign(v, p) for v, p in zip(self.model.trainable_variables, self._placeholders)]
        self._assign_op = tf.group(*assigns)

        self._gradients_placeholders = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape())
                                        for v in self.model.trainable_variables]
        self.apply_metagradients = self.optimizer.apply_gradients(zip(self._gradients_placeholders,
                                                                      self.model.trainable_variables))
        # clipped_grads = [(tf.clip_by_norm(grad, 10), var) for grad, var in zip(self._gradients_placeholders,
        #                                                                        self.model.trainable_variables)]
        # self.apply_metagradients = self.optimizer.apply_gradients(clipped_grads)

        ## Create and add an explicit regularizer for the model (L2 distance from the starting weights)
        self.regularizer_initial_params = [tf.Variable(tf.zeros(v.get_shape(), dtype=v.dtype.base_dtype))
                                           for v in model.trainable_variables]
        self.assign_regularizer_ops = tf.group(*[tf.assign(v, p) for v, p in zip(self.regularizer_initial_params, self.model.trainable_variables)]) # must call this AFTER the initial parameters have been reset!
        def l2_loss_phi_phi0():
            aux_loss = tf.add_n([ tf.reduce_sum( \
                                  tf.square(model.trainable_variables[i] - self.regularizer_initial_params[i]))
                                for i in range(len(model.trainable_variables))])
            aux_loss = tf.identity(0.5 * self.lambda_reg * aux_loss, name='iMAML-Regularizer')
            return aux_loss
        # Remove the previous regularizer, if present (this is to guarantee that the correct variables are used
        # for regularization).
        ## model.losses = [l for l in model.losses if not l.name.startswith('iMAML-Regularizer')]
        if len([l for l in model.losses if l.name.startswith('iMAML-Regularizer')])>0:
            # Most likely because of model saving/loading or wrapping it in a meta-learner and then into
            # another one
            print("iMAML ERROR: adding regularizer multiple times!")
        model.add_loss(l2_loss_phi_phi0)



    def _tf_list_of_tensors_to_vector(self, weights):
        w_vector = None
        for w in weights:
            w_unwrap = tf.reshape(w, [-1])
            if w_vector is None:
                w_vector = tf.identity(w_unwrap)
            else:
                w_vector = tf.concat([w_vector, w_unwrap], axis=0)
        return tf.reshape(w_vector,[-1])

    def _tf_vector_to_list_of_tensors(self, weights, w_vec):
        tensors = []
        begin_index = 0
        for w in weights:
            shape = w.get_shape()
            w_size = 1
            for dim in shape:
                w_size *= dim.value

            w_tensor = tf.reshape(tf.slice(w_vec,begin=[begin_index],size=[w_size]), w.shape)
            begin_index += w_size

            tensors.append(w_tensor)
        return tensors

    def _gradients_for_task(self, t):
        """
        Hacky utility function to compute the iMAML meta-gradient for a task t.
        For the moment, this only works for Keras models with a single input.

        The gradient is approximated by optimization (Eq. (7) of the iMAML paper).
        """
        test_x, test_y = t.get_test_set()
        train_x, train_y = t.get_train_set() if len(test_y)>len(t.train_indices) else t.sample_batch(len(test_y))


        # 1) Build the computational graph to compute the Hessian-vector product (\nabla^2_\phi \hat{L}_i(\phi_i) * w)
        #    + build the computational graph to compute the test-set gradients (\nabla_\phi L_i(\phi_i))
        if self.target_placeholder is None:
            # Only build the graph on the first time this function is called
            # `model' is run through self.model.inputs[0] to self.model.output

            self.target_placeholder = tf.placeholder(tf.float32, [None, ] + list(test_y.shape[1:]),
                                                     name="imaml_target_test")
            self.loss = tf.reduce_mean(self.model.loss(self.target_placeholder, self.model.output))
            self.gradients_ = tf.gradients(self.loss, self.model.trainable_variables)

            # self.gradients_ can now be used to compute both the test-set gradients and the (train-set!) Hessian-vector products
            self.Hw = None
            self.G = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape())
                                  for v in self.model.trainable_variables]

            # 2) Evaluate the value of the function to minimize and its derivative
            def func(x):
                # We wish to minimize the following function (Eq. (7) of iMAML paper)
                # x^T * ( x + 1/lambda * [Hessian-vector product between
                #   "hessian of TRAIN loss at phi_i" and "x"] ) - x^T * "gradient of TEST loss at phi_i"
                # = x^T * (x + 1/lambda * Hw) - x^T * G
                # Where Hw is obtained using "self._hessian_vector_prod", setting "self._placeholders" to
                # the current value of "x", and
                #       {self.model.inputs[0]: TRAIN_batch_x, self.target_placeholder: TRAIN_batch_y})
                # in order to compute gradients on the training loss
                # Where G is computed by running (once; the value is constant wrt to x!) self._gradients_
                # using {self.model.inputs[0]: TESt_batch_x, self.target_placeholder: TEST_batch_y})
                #     G is computed before running the optimization process on func(), and is stored
                #     in self.last_test_gradients

                # convert x from a single vector to a list of tensors, x_vars
                x_vars = self._tf_vector_to_list_of_tensors(self.model.trainable_variables, x)

                if self.Hw is None:
                    # Compute the Hessian-vector product
                    self._grad_dot_vector = tf.add_n( [tf.reduce_sum(self.gradients_[i] * x_vars[i]) for i in range(len(x_vars))] )
                    self.Hw = tf.gradients(self._grad_dot_vector, self.model.trainable_variables)

                # Value = wT * (w + 1/lambda * Hw - G)
                value = tf.add_n( [tf.reduce_sum(x_vars[i] * (x_vars[i] + 1.0/self.lambda_reg*self.Hw[i] - self.G[i]) )
                                    for i in range(len(self.model.trainable_variables))] )

                # Gradient = 2*w + 2/lambda*Hw - G
                gradients = [2*x_vars[i] + 2.0/self.lambda_reg*self.Hw[i] - self.G[i]
                             for i in range(len(self.model.trainable_variables))]

                # linearize gradients into a single vector!
                gradient = self._tf_list_of_tensors_to_vector(gradients)

                return value, gradient

            self._optim_func = func

            # 3) Approximate the gradient with a fixed number of optimization steps
            n = 0
            for w in self.model.trainable_variables:
                shape = w.get_shape()
                w_size = 1
                for dim in shape:
                    w_size *= dim.value
                n += w_size
            #"""
            self.x_op = tfp.optimizer.lbfgs_minimize(
                self._optim_func,
                tf.zeros((n,), dtype=tf.float32),
                num_correction_pairs=10,
                tolerance=1e-08,
                max_iterations=self.n_iters_optimizer,
                parallel_iterations=1,
            )
            #"""

            self.result_vector_placeholder = tf.placeholder(dtype=tf.float32, shape=(n,))
            self.result_to_list_of_tensors = self._tf_vector_to_list_of_tensors(
                                                        self.model.trainable_variables,
                                                        self.result_vector_placeholder)


        # Perform optimization

        # Only need to evaluate this once, as it is constant during the optimization process
        self.last_test_gradients = self.session.run(self.gradients_,
                                                    {self.model.inputs[0]: test_x, self.target_placeholder: test_y})

        args = dict(zip(self.G, self.last_test_gradients))
        args[self.model.inputs[0]] = train_x
        args[self.target_placeholder] = train_y
        result = self.session.run(self.x_op, args)
        #print(result.num_iterations)
        #print(result.objective_value)
        #print(np.sum(np.sum(result.objective_gradient)))

        approx_metagradient = self.session.run(self.result_to_list_of_tensors, {self.result_vector_placeholder:result.position})

        return approx_metagradient

    def initialize(self, session):
        """
        This method should be called after tf.global_variables_initializer().
        """
        self.session = session
        self.current_initial_parameters = self.session.run(self.model.trainable_variables)

    def task_begin(self, task=None, **kwargs):
        """
        Method to be called before training on each meta-batch task
        """
        super().task_begin(task=task)

        # Reset the model to the current weights initialization
        self.session.run(self._assign_op, feed_dict=dict(zip(self._placeholders, self.current_initial_parameters)))
        self.session.run(self.assign_regularizer_ops) # Only run this AFTER the above (which assigns the
                                                      # current_initial_parameters back to the model parameters

    def task_end(self, task=None, **kwargs):
        """
        Method to call after training on each meta-batch task; possibly return relevant information for the
        meta-learner to use for the meta-updates.
        """
        assert task is not None, "FOMAML needs a `task' argument on .task_end to compute the data it needs."

        # Compute the gradient of the TEST LOSS evaluated at the final parameters. E.g., mean gradients
        # over a batch of test data

        if isinstance(task, TaskAsSequenceOfTasks):
            # Default: evaluate performance on the last task only
            ret_grads = self._gradients_for_task(task.get_task_by_index(-1))
        else:
            ret_grads = self._gradients_for_task(task)

        return ret_grads

    def update(self, list_of_final_gradients, **kwargs):
        """
        Main (outer-loop) training operation. Call after training on a whole meta-batch, after each meta-iteration
        has finished.
        """
        # Perform a FOMAML update for the outer meta-training iteration. Take the expected test gradient
        # L'(\tilde{phi}) over tasks in the meta-batch and perform a single step of gradient descent.
        avg_final_grads = []
        for grads in zip(*list_of_final_gradients):
            avg_final_grads.append(np.mean(grads, axis=0))

        # Apply gradients to the initial parameters
        self.session.run(self._assign_op, feed_dict=dict(zip(self._placeholders, self.current_initial_parameters)))
        self.session.run(self.apply_metagradients, feed_dict=dict(zip(self._gradients_placeholders, avg_final_grads)))

        self.current_initial_parameters = self.session.run(self.model.trainable_variables)
