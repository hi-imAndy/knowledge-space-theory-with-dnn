import numpy as np
from data.data_loader import load_test_dataset
from tensorflow import keras
from kst.kst import create_ks
from utils import log_to_file, get_models_path


def get_predict_results():
    test_ds = load_test_dataset()
    models = load_models()
    val_indexes = get_val_index_for_each_class(test_ds)
    predictions = {}
    for model_idx, model in enumerate(models):
        predictions_y = model.predict(test_ds)
        predicted_classes = np.argmax(predictions_y, axis=1)
        prediction = []
        for i, idx in enumerate(val_indexes):
            prediction.append(1 if predicted_classes[idx] == i else 0)
        predictions['model' + str(model_idx)] = prediction
    log_to_file(predictions)
    return predictions


def get_val_index_for_each_class(ds):
    indexes = []
    for _, labels in ds.take(1):
        labels = labels.numpy().tolist()
        for i in range(0, 3):
            indexes.append(labels.index(i))
    return indexes


def load_models():
    models = []
    for i in range(0, 500):
        try:
            path = get_models_path() + f'model{i}'
            model = keras.models.load_model(path)
            models.append(model)
        except Exception as e:
            break
    return models


def convert_to_kst_dict(results: dict):
    kst_results = {
        'ellipse': [],
        'square': [],
        'triangle': []
    }
    for res in results.values():
        for i, key in enumerate(kst_results.keys()):
            kst_results[key].append(res[i])
    return kst_results


if __name__ == "__main__":
    results = get_predict_results()
    kst_results = convert_to_kst_dict(results)
    log_to_file(kst_results)
    ks = create_ks(kst_results)
    log_to_file(ks)
