import copy

import matplotlib.pyplot as plt
import numpy as np

from data.data_loader import load_dataset
from dnn.model import setup_models, setup_easy_model
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
        model.save(f'{model_path}model{model_idx + 149}', save_format='h5')


def train_curriculum_easy():
    model = setup_easy_model()
    train_ds, val_ds = load_dataset()
    ks = [2, 1, 0]
    for i, state in enumerate(ks):
        train_x = np.concatenate([x for x, y in train_ds], axis=0)
        train_y = np.concatenate([y for x, y in train_ds], axis=0)
        val_x = np.concatenate([x for x, y in val_ds], axis=0)
        val_y = np.concatenate([y for x, y in val_ds], axis=0)
        history = model.fit(train_ds, epochs=100, validation_data=val_ds)
        print('trained')


if __name__ == "__main__":
    train_models()
