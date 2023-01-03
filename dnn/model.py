import random
import tensorflow as tf


def setup_models():
    models = []
    for i in range(0, 500):
        model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1. / 255),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D()
        ])
        additional_cnn_layers = random.randrange(0, 3)
        for j in range(0, additional_cnn_layers):
            model.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D())
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(3, activation='sigmoid'))

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        models.append(model)
    return models
