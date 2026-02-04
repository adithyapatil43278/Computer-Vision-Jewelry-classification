import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

def get_datasets(train_dir, test_dir, img_size, batch_size, seed):

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    def prepare(ds, training=False):
        ds = ds.map(
            lambda x, y: (tf.keras.applications.efficientnet.preprocess_input(x),
                          tf.one_hot(y, depth=num_classes)),
            num_parallel_calls=AUTOTUNE
        )
        if training:
            ds = ds.shuffle(1000)
        return ds.prefetch(AUTOTUNE)

    return (
        prepare(train_ds, True),
        prepare(val_ds),
        prepare(test_ds),
        class_names,
        num_classes,
        augmentation
    )
