import tensorflow as tf

LOSS = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

def train(model, train_ds, val_ds, epochs):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=LOSS,
        metrics=["accuracy"]
    )

    h1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    return h1


def finetune(model, base_model, train_ds, val_ds, epochs):

    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=LOSS,
        metrics=["accuracy"]
    )

    h2 = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    return h2
