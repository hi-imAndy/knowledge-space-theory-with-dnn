import joblib
import matplotlib.pyplot as plt
import numpy as np
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


def get_val_index_for_each_class(ds):
    indexes = []
    for _, labels in ds.take(1):
        labels = labels.numpy().tolist()
        for i in range(0, 3):
            indexes.append(labels.index(i))
    return indexes


def train_models():
    models = setup_models()
    train_ds, val_ds = load_dataset()
    val_indexes = get_val_index_for_each_class(val_ds)
    predictions = {}
    model_path = get_models_path()
    for model_idx, model in enumerate(models):
        history = model.fit(train_ds, epochs=20, validation_data=val_ds)
        joblib.dump(model, f'{model_path}model{model_idx}.joblib')
        predictions_y = model.predict(val_ds)
        predicted_classes = np.argmax(predictions_y, axis=1)
        prediction = []
        for i, idx in enumerate(val_indexes):
            prediction.append(1 if predicted_classes[idx] == i else 0)
        predictions['model' + str(model_idx)] = prediction
    log_to_file(predictions)
    return predictions

