import tensorflow as tf

def build_model(base_model, augmentation, img_size, num_classes):

    inputs = tf.keras.layers.Input(shape=img_size + (3,))
    x = augmentation(inputs, training=True)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.models.Model(inputs, outputs)
