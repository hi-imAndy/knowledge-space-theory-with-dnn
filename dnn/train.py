import matplotlib.pyplot as plt
from data.data_loader import load_dataset
from dnn.model import setup_models
from utils import *


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
    model_path = get_models_path()
    for model_idx, model in enumerate(models):
        model.fit(train_ds, epochs=32, validation_data=val_ds)
        model.save(f'{model_path}model{model_idx + 140}', save_format='h5')


if __name__ == "__main__":
    train_models()
