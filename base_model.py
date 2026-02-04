import tensorflow as tf

def get_base_model(img_size):
    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=img_size + (3,)
    )

    base_model.trainable = False
    return base_model
