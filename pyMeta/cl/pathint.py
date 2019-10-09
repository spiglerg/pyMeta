"""
Implementation of Path Integral / Synaptic Intelligence, Zenke et. al (2017).

- Add a wrapper to a tf.keras.Model to add useful variables, operations and callbacks.

Example:

# Decoration
model = --your tf.keras.Model--
model = PathIntSGD(param_c=0.1, param_xi=0.1)(model)

# Call the following after training on each task:
model.run_after_task()


NOTE: if you want to run the same model on multiple sequences of tasks, you should manually reset the state
of the cl algorithm with
model.cl_reset()

NOTE2: if you use .fit_n_iterations on TaskAsSequenceOfTasks, this is done automatically.
But if you prefer to run each task individually, you should remember to call the run_after_task callback.

NOTE3: when wrapping a model loaded with tf.keras.models.load_model, make sure you set compile=False, and then
re-compile the model yourself.
"""
# TODO:

import tensorflow as tf



def PathIntSGD(param_c=0.1, param_xi=0.1):
    """
    Simplified implementation of Path Integral that only works when training using SGD.
    This allows to compute the gradients at the previous step as \delta W / learning_rate,
    without having to explicitly recompute them.
    """
    def modify_model(model):
        # Add new Variables
        model._pathint = {}
        model._pathint['small_omega']    = [tf.Variable(tf.zeros(v.get_shape(), dtype=v.dtype.base_dtype))
                                            for v in model.trainable_variables]
        model._pathint['big_omega']      = [tf.Variable(tf.zeros(v.get_shape(), dtype=v.dtype.base_dtype))
                                            for v in model.trainable_variables]
        model._pathint['target_weights'] = [tf.Variable(tf.zeros(v.get_shape(), dtype=v.dtype.base_dtype))
                                            for v in model.trainable_variables]
        model._pathint['prev_weights']   = [tf.Variable(tf.zeros(v.get_shape(), dtype=v.dtype.base_dtype))
                                            for v in model.trainable_variables]

        model._pathint['update_ops'] = []

        model._pathint['reset'] = tf.group( tf.initialize_variables(model._pathint['small_omega']+model._pathint['big_omega']+model._pathint['target_weights']+model._pathint['prev_weights']), [tf.assign(model._pathint['prev_weights'][i], model.trainable_variables[i]) for i in range(len(model.trainable_variables))] )


        def cl_reset():
            tf.get_default_session().run( model._pathint['reset'] )
        model.cl_reset = cl_reset
        if tf.get_default_session() is not None:
            model.cl_reset()

        # Add the PathInt quadratic regularizer
        def path_int_auxiliary_loss():
            aux_loss = tf.add_n([ tf.reduce_sum(model._pathint['big_omega'][i] * \
                                  tf.square(model.trainable_variables[i] - model._pathint['target_weights'][i]))
                                for i in range(len(model.trainable_variables))])
            aux_loss *= param_c
            return aux_loss
        model.add_loss(path_int_auxiliary_loss)


        def add_update_ops():
            # Add relevant update-ops
            for op in model._pathint['update_ops']:
                model.add_update(op)

            lr = model.optimizer.get_config()['learning_rate']

            update_small_omega_ops = [tf.assign_add(model._pathint['small_omega'][i],
                                                    tf.square(model.trainable_variables[i] - model._pathint['prev_weights'][i]) / lr )
                                      for i in range(len(model.trainable_variables)) ]
            update_small_omega_op = tf.group(*update_small_omega_ops)
            model.add_update(update_small_omega_op)

            with tf.control_dependencies([update_small_omega_op]):
                # Update last weights, for computation of theta^{t}-theta^{t-1}
                update_prev_weights_ops = [tf.assign(model._pathint['prev_weights'][i],
                                                     model.trainable_variables[i])
                                           for i in range(len(model.trainable_variables)) ]
                update_prev_weights_op = tf.group(*update_prev_weights_ops)
                model.add_update(update_prev_weights_op)


        if hasattr(model, 'optimizer') and model.optimizer is not None:
            add_update_ops()
        else:
            def new_compile(*args, **kwargs):
                model._old_compile(*args, **kwargs)
                add_update_ops()

            model._old_compile = model.compile
            model.compile = new_compile


        # Add callbacks
        update_big_omega_ops1 = []
        update_big_omega_ops2 = []
        update_big_omega_ops3 = []
        for i in range(len(model.trainable_variables)):
            update_big_omega_ops1.append(tf.assign_add(model._pathint['big_omega'][i],
                                                       tf.div(model._pathint['small_omega'][i],
                                                              param_xi + tf.square(model.trainable_variables[i] - model._pathint['target_weights'][i]) ) ))

            update_big_omega_ops2.append(tf.assign(model._pathint['small_omega'][i],
                                                   0.0*model._pathint['small_omega'][i]))

            update_big_omega_ops3.append( tf.assign(model._pathint['target_weights'][i],
                                                    model.trainable_variables[i]) )

        # Update big omega variables
        model._pathint['update_big_omega_op1'] = tf.group(*update_big_omega_ops1)
        # Reset small omega variables to zero
        model._pathint['update_big_omega_op2'] = tf.group(*update_big_omega_ops2)
        # Update the target weights
        model._pathint['update_big_omega_op3'] = tf.group(*update_big_omega_ops3)

        def run_after_task():
            session = tf.get_default_session()
            session.run(model._pathint['update_big_omega_op1'])
            session.run(model._pathint['update_big_omega_op2'])
            session.run(model._pathint['update_big_omega_op3'])
        model.run_after_task = run_after_task


        return model
    return modify_model  




