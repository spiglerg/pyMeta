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
from copy import copy

from scipy.sparse.linalg import cg, LinearOperator

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


    def _gradients_for_task_CG(self, t):
        """
        Hacky utility function to compute the iMAML meta-gradient for a task t.
        For the moment, this only works for Keras models with a single input.

        The gradient is approximated by optimization (Eq. (7) of the iMAML paper).

        Optimization is performed using the Conjugate Gradient method.


        Solve the problem:
            argmin_w || w - (I+1/lambda*H)*g ||
        where H is the Hessian matrix of a function (here, training loss for the inner-loop optimization, at the final
        parameters) whose gradient is h, and g is a vector (here, the test loss for the inner-loop optimization, at the final parameters)

        We use the Conjugate Gradient method and we compute Hessian-vector products using Pearlmutter's method
        (Pearlmutter (1994)), Hv = D(h*v), where H=D(h) and D(v)=0.

        Solving the above problem is equivalent to solving the linear system
        (I+1/lambda*H) * w = g

        We modify the CG method slightly to exploit the efficient Hessian-vector products, thus without ever explicitly
        computing or storing the full matrix 'A':
            A = (I+1/lambda*H)
            b = g

        Hv is a HvProduct object that when called with a vector as argument (of the correct size), it returns the correct
        Hessian-vector product.

        """
        test_x, test_y = t.get_test_set()
        train_x, train_y = t.get_train_set() if len(test_y)>len(t.train_indices) else t.sample_batch(len(test_y))


        # 1) Build the computational graph to compute the Hessian-vector product (\nabla^2_\phi \hat{L}_i(\phi_i) * w)
        #    + build the computational graph to compute the test-set gradients (\nabla_\phi L_i(\phi_i))
        if not hasattr(self, '_CG_graph_initialized'):
            # Only build the graph on the first time this function is called
            # `model' is run through self.model.inputs[0] to self.model.output

            self._CG_graph_initialized = True

            self.target_placeholder = tf.placeholder(tf.float32, [None, ] + list(test_y.shape[1:]),
                                                     name="imaml_target_test")
            self.loss = tf.reduce_mean(self.model.loss(self.target_placeholder, self.model.output))
            self.gradients_ = tf.gradients(self.loss, self.model.trainable_variables)

            # Compute the Hessian-vector product (current constant vector should be passed via self._placeholders)
            self._grad_dot_vector = tf.add_n( [tf.reduce_sum(self.gradients_[i] * self._placeholders[i]) for i in range(len(self._placeholders))] )
            self.Hv = tf.gradients(self._grad_dot_vector, self.model.trainable_variables)


        # Perform optimization

        # Only need to evaluate this once, as it is constant during the optimization process
        test_gradients = self.session.run(self.gradients_,
                                          {self.model.inputs[0]: test_x, self.target_placeholder: test_y})

        # Conjugate gradient
        # 0-iterations of CG should return the FOMAML gradient + we expect the solution to be near there
        x = [copy(g) for g in test_gradients] #test_gradients[:] #[np.ones(var.shape) for var in self.current_initial_parameters]
        #x = [np.zeros(var.shape) for var in self.current_initial_parameters]

        b = test_gradients


        def Av(v):
            args = dict(zip(self._placeholders, v))
            args[self.model.inputs[0]] = train_x
            args[self.target_placeholder] = train_y
            Hvp = self.session.run(self.Hv, args)
            return [v[i] + 1.0/self.lambda_reg * Hvp[i] for i in range(len(v))]

        Ax = Av(x)

        r = [b[i] - Ax[i] for i in range(len(x))]
        p = r[:]
        for i in range(self.n_iters_optimizer):
            Ap = Av(p)

            #alpha = np.sum(r*r) / np.sum(p*Ap)
            alpha = np.sum([ np.sum(r[i]*r[i]) for i in range(len(x)) ]) / np.sum([ np.sum(p[i]*Ap[i]) for i in range(len(x)) ])

            x = [x[i] + alpha*p[i] for i in range(len(x))]
            rnew = [r[i] - alpha*Ap[i] for i in range(len(x))]

            # if r is small enough, exit loop
            #if np.sqrt(np.sum(rnew*rnew)) < 1e-10:
            #    break

            #beta = np.sum(rnew*rnew) / np.sum(r*r)
            beta = np.sum([ np.sum(rnew[i]*rnew[i]) for i in range(len(x)) ]) / np.sum([ np.sum(r[i]*r[i]) for i in range(len(x)) ])

            p = [rnew[i] + beta*p[i] for i in range(len(x))]
            r = rnew

        return x

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
            #ret_grads = self._gradients_for_task(task.get_task_by_index(-1))
            ret_grads = self._gradients_for_task_CG(task.get_task_by_index(-1))
        else:
            #ret_grads = self._gradients_for_task(task)
            ret_grads = self._gradients_for_task_CG(task)

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
