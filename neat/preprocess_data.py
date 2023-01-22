import numpy as np

from data.data_loader import load_dataset
import tensorflow as tf

if __name__ == '__main__':
    train_ds, val_ds = load_dataset()
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    images, labels = train_ds
    images = images / 255.0
    labels = labels.astype(np.int32)
    print('')
