import tensorflow as tf
import numpy as np

CLASS_NAMES = [
    'o', 'ki', 'su', 'tsu', 'na',
    'ha', 'ma', 'ya', 're', 'wo'
]

def set_seed(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)

def load_kmnist(batch_size=64, seed=42):
    set_seed(seed)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.kmnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    val_size = 10000
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]

    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_train = ds_train.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    ds_val = ds_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    ds_test = ds_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_test, CLASS_NAMES
