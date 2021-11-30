"""
Pre-defined network models.
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense, Activation, \
                                    GlobalAveragePooling2D


"""
# Old, working (before multiheaded options
def make_omniglot_cnn_model(num_output_classes):
    model = tf.keras.models.Sequential()
    for i in range(4):
        if i == 0:
            model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation=None,
                             input_shape=[28, 28, 1]))

        else:
            model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation=None))

        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Activation('relu'))

    model.add(Flatten())
    # model.add(GlobalAveragePooling2D())
    # model.add(Dense(256, activation=tf.nn.relu))
    model.add(Dense(num_output_classes, activation='softmax'))

    return model
"""

def make_omniglot_cnn_model(num_output_classes, multi_headed=False, num_heads=-1, input_shape=(28,28,1)):
    inputs = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32)
    x = inputs
    for i in range(4):
        x = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation=None)(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=2, strides=2)(x) #, padding="same")(x)
        x = Activation('relu')(x)

    x = Flatten()(x)
    # model.add(GlobalAveragePooling2D())
    # x = Dense(256, activation=tf.nn.relu)(x)

    if not multi_headed:
        x = Dense(num_output_classes, activation='softmax')(x)
    else:
        # Create 'num_heads' heads
        heads = []
        for i in range(num_heads):
            heads.append( Dense(num_output_classes, activation='softmax', name='multihead_'+str(i))(x) )
        x = tf.keras.layers.concatenate(heads)

    model = tf.keras.models.Model(inputs=inputs, outputs=x) #[a, b])

    return model




def make_miniimagenet_cnn_model(num_output_classes, multi_headed=False, num_heads=-1, input_shape=(84,84,3)):
    inputs = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32)
    x = inputs
    for i in range(4):
        x = Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation=None)(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=2, strides=2, padding="same")(x)
        x = Activation('relu')(x)

    x = Flatten()(x)
    # model.add(GlobalAveragePooling2D())
    # x = Dense(256, activation=tf.nn.relu)(x)

    if not multi_headed:
        x = Dense(num_output_classes, activation='softmax')(x)
    else:
        if num_heads==1:
            num_heads=2

        # Create 'num_heads' heads
        heads = []
        for i in range(num_heads):
            heads.append( Dense(num_output_classes, activation='softmax', name='multihead_'+str(i))(x) )
        x = tf.keras.layers.concatenate(heads)

    model = tf.keras.models.Model(inputs=inputs, outputs=x) #[a, b])
    return model


def make_core50_cnn_model(num_output_classes):
    model = tf.keras.models.Sequential()
    for i in range(4):
        if i == 0:
            model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation=None,
                             input_shape=[128, 128, 3]))
        else:
            model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation=None))

        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=2, strides=2, padding="same"))
        model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(num_output_classes, activation='softmax'))

    return model


def make_sinusoid_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[1]),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    return model



# Set of callbacks to pass to SeqFOMAML when a multi-headed CNN is used.
def multihead_callback_re_init(model):
    # Resetr all weights with 'multihead' in the name to their initial value
    for layer in model.layers:
        if layer.name.find('multihead') >= 0:
            layer.bias.assign( layer.bias_initializer(layer.bias.shape) )
            layer.kernel.assign( layer.kernel_initializer(layer.kernel.shape) )
def multihead_callback_copy_head0(model):
    # Copy all weights from the first head (note: before fine-tuning it!)
    # This works much better than random re-init, at least early during meta-training.
    # Fully trained models may not differ much in performance.
    layer_0 = [layer for layer in model.layers if layer.name=="multihead_0"]
    if len(layer_0)<1:
        return
    layer_0 = layer_0[0]
    for layer in model.layers:
        if layer.name.find('multihead') >= 0 and layer.name != "multihead_0":
            layer.bias.assign( layer_0.bias )
            layer.kernel.assign( layer_0.kernel )

