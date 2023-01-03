import matplotlib.pyplot as plt
from data.data_loader import load_dataset
import numpy as np

from dnn.model import setup_models

class_names = ['ellipse', 'square', 'triangle']


def plot_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')


def train_models():
    models = setup_models()
    train_ds, val_ds = load_dataset()
    for model in models:
        history = model.fit(train_ds, epochs=10, validation_data=val_ds)
        predictions = model.predict(val_ds)
        predicted_classes = np.argmax(predictions, axis=1)
        plot_history(history)
        model.summary()

