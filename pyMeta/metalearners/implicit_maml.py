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
from copy import copy, deepcopy

from scipy.sparse.linalg import cg, LinearOperator

from pyMeta.core.meta_learner import GradBasedMetaLearner
from pyMeta.core.task import TaskAsSequenceOfTasks


#from pyMeta.metalearners.fomaml import grads_on_batch
@tf.function
def grads_on_batch(model, batch_X, batch_y):
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(model.loss(batch_y, model(batch_X, training=True)))
    grads = tape.gradient(loss, model.trainable_variables)
    return grads

@tf.function
def Hvp(model, batch_X, batch_y, v):
    """
    Computes an Hessian-vector product using the Pearlmutter method.
    The Hessian used is the loss function of tf.keras.Model 'model' with respect to its trainable_variables.
    The loss is evaluated on the given batch.
    The vector v is decomposed into a list of Tensors, to match the number of weights of 'model' and their shape.
    """
    with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
            loss = tf.reduce_mean(model.loss(batch_y, model(batch_X, training=True)))
        grads = inner_tape.gradient(loss, model.trainable_variables)
    Hv = outer_tape.gradient(grads, model.trainable_variables, output_gradients=v)

    return Hv


# Debugging functions to print the value of 2 functionals
def f1(x, Av, b):
    Ax = Av(x)
    return 0.5 * np.sum([ np.sum(x[k]*(Ax[k]-b[k])) for k in range(len(x))])
def f2(x, Av, b):
    Ax = Av(x)
    return 0.5 * np.sqrt(np.sum([ np.sum(np.square(Ax[k]-b[k])) for k in range(len(x))]))


# Optimizers that can be used to approximate the meta-gradient (I+1/lambda*H)*g = b -> Ag = b
# A is supplied as a linear operator function, taking a vector as argument and returning the matrix-vector product.
def plain_gradient_descent(Av, b, x0, num_iterations, learning_rate=0.01, debug=False):
    ## Gradient descent
    for i in range(num_iterations):
        Ax = Av(x0)
        for k in range(len(x0)):
            x0[k] = x0[k] - 0.01 * (Ax[k]-b[k])

        if debug:
            print(f1(x0, Av, b), f2(x0, Av, b) , '\n\n')

    return x0

def steepest_descent(Av, b, x0, num_iterations, debug=False):
    ## Steepest descent + line search
    Ax = Av(x0)
    r = [b[i] - Ax[i] for i in range(len(x0))]
    for i in range(num_iterations):
        rTr = np.sum([ np.sum(r[k]*r[k]) for k in range(len(x0)) ])
        Ar = Av(r)
        alpha = rTr / np.sum([ np.sum(r[k]*Ar[k]) for k in range(len(x0)) ])

        x0 = [x0[k] + alpha*r[k] for k in range(len(x0))]
        r = [r[k] - alpha*Ar[k] for k in range(len(x0))]

        if debug:
            print(f1(x0, Av, b), f2(x0, Av, b) , '\n\n')

    return x0

def conjugate_gradient(Av, b, x0, num_iterations, debug=False):
    ## Conjugate gradient
    Ax = Av(x0)

    r = [b[i] - Ax[i] for i in range(len(x0))]
    p = deepcopy(r)
    for i in range(num_iterations):
        Ap = Av(p)

        #alpha = np.sum(r*r) / np.sum(p*Ap)
        rTr = np.sum([ np.sum(r[k]*r[k]) for k in range(len(x0)) ])
        alpha = rTr / np.sum([ np.sum(p[k]*Ap[k]) for k in range(len(x0)) ])

        x0 = [x0[k] + alpha*p[k] for k in range(len(x0))]
        r = [r[k] - alpha*Ap[k] for k in range(len(x0))]

        # if r is small enough, exit loop
        #if np.sqrt(np.sum(rnew*rnew)) < 1e-10:
        #    break

        #beta = np.sum(rnew*rnew) / np.sum(r*r)
        beta = np.sum([ np.sum(r[k]*r[k]) for k in range(len(x0)) ]) / rTr

        p = [r[k] + beta*p[k] for k in range(len(x0))]

        if debug:
            print(f1(x0, Av, b), f2(x0, Av, b), (alpha>0), '\n\n')

    return x0




class iMAMLMetaLearner(GradBasedMetaLearner):
    def __init__(self, model, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), lambda_reg=1.0, n_iters_optimizer=5, name="FOMAMLMetaLearner"):
        """
        In general, meta-learners objects should be created before the tf.keras.Model that they wrap is compiled,
        in case losses or regularizers need to be added.

        This meta-learner should be used as follows:
        + Instantiate object
        [Compile the wrapped model]
        + Initialize object
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

        model.imaml_reg_initial_params = [tf.Variable(tf.zeros(v.shape),
                                                      dtype=v.dtype.base_dtype,
                                                      trainable=False,
                                                      name="iMAML_regularizer_initial_parameters")
                                           for v in model.trainable_variables]

        def imaml_regularizer():
            reg = tf.add_n([ tf.reduce_sum(tf.square(model.trainable_variables[i] - model.imaml_reg_initial_params[i]))
                            for i in range(len(model.trainable_variables))])
            return 0.5 * self.lambda_reg * reg
        self.model.add_loss(imaml_regularizer)


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
        # Compute the meta-gradient by solving the corresponding optimization problem

        test_x, test_y = t.get_test_set()
        train_x, train_y = t.get_train_set() #t.get_train_set() if len(t.train_indices) > len(test_y) else t.sample_batch(len(test_y))



        # Linear operator for conjugate gradient, wrapping the Hessian-vector product.
        def Av(v):
            Hv = Hvp(self.model, train_x, train_y, v)
            return [v[i] + 1.0/self.lambda_reg * Hv[i].numpy() for i in range(len(v))]


        test_gradients = [g.numpy() for g in grads_on_batch(self.model, test_x, test_y)]

        #x0 = deepcopy(test_gradients)
        x0 = [np.zeros(var.shape, dtype=np.float32) for var in self.current_initial_parameters]

        b = deepcopy(test_gradients)




        debug = False
        #debug = True
        #self.n_iters_optimizer = 10


        #x = conjugate_gradient(Av, b, x0, self.n_iters_optimizer, debug=debug)
        x = steepest_descent(Av, b, x0, self.n_iters_optimizer, debug=debug)
        #x = plain_gradient_descent(Av, b, x0, self.n_iters_optimizer, learning_rate=0.01, debug=debug)

        # lambda=2.0?


        """
        # vonNeumann approximation of the inverse of (I+1/lambda*H)^{-1}
        x = deepcopy(b) # j=0:  (-1/l*H)^j * v = v 
        prev_v = deepcopy(b)
        for j in range(1, 5):
            Hv = Hvp(self.model, train_x, train_y, prev_v)
            prev_v = [-1.0/self.lambda_reg * Hv[i].numpy() for i in range(len(b))]
            x = [x[i] + prev_v[i] for i in range(len(b))]
        #"""



        if debug:
            print(list(zip(x[-1], test_gradients[-1], Av(x)[-1])))
            print("*****\n")
            import sys
            sys.exit()


        return x

    def initialize(self):
        """
        This method should be called after the wrapped model is compiled.
        """
        self.current_initial_parameters = [v.numpy() for v in self.model.trainable_variables]

    def task_begin(self, task=None, **kwargs):
        """
        Method to be called before training on each meta-batch task
        """
        super().task_begin(task=task)

        # Reset the model to the current weights initialization
        for i in range(len(self.current_initial_parameters)):
            self.model.trainable_variables[i].assign( self.current_initial_parameters[i] )
            self.model.imaml_reg_initial_params[i].assign( self.current_initial_parameters[i] )

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
            ret_grads = self._gradients_for_task_CG(task.get_task_by_index(-1))
        else:
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

        # Apply gradients to the *initial parameters*
        for i in range(len(self.current_initial_parameters)):
            self.model.trainable_variables[i].assign( self.current_initial_parameters[i] )

        self.optimizer.apply_gradients(zip(avg_final_grads, self.model.trainable_variables))

        # Set the new initial parameters
        self.current_initial_parameters = [v.numpy() for v in self.model.trainable_variables]
