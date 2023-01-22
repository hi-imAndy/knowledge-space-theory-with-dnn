import glob

import numpy as np
from keras.utils import image_dataset_from_directory
from PIL import Image
from sklearn.utils import shuffle

train_url = '../../generated_images/dataset'
test_url = '../../generated_images_test/dataset'
IMG_SIZE = 96
BATCH_SIZE = 500

CLASSES = ['ellipse', 'ellipse_curved', 'ellipse_hard', 'ellipse_line', 'square', 'square_curved', 'square_hard',
           'square_line', 'triangle', 'triangle_curved', 'triangle_hard', 'triangle_line']


def load_dataset():
    train_ds = image_dataset_from_directory(
        validation_split=0.2, subset="training", seed=123,
        directory=train_url, labels='inferred', batch_size=BATCH_SIZE,
        image_size=(IMG_SIZE, IMG_SIZE))
    val_ds = image_dataset_from_directory(
        validation_split=0.2, subset="validation", seed=123,
        directory=train_url, labels='inferred', batch_size=BATCH_SIZE,
        image_size=(IMG_SIZE, IMG_SIZE))
    return train_ds, val_ds


def load_test_dataset():
    test_ds = image_dataset_from_directory(
        seed=123, directory=test_url, labels='inferred',
        batch_size=BATCH_SIZE, image_size=(IMG_SIZE, IMG_SIZE))
    return test_ds


def load_data_by_label(label):
    data = []
    for filename in glob.glob(f'{train_url}/{label}/*.png'):
        im = Image.open(filename).convert('L')
        data.append(np.array(im))
    data_np = np.array(data)
    return data_np


def load_other_classes(classes: list):
    data = []
    for cl in CLASSES:
        if cl not in classes:
            class_data = load_data_by_label(cl)
            data.extend(list(class_data))
    return data


def load_by_class(label: list):
    x = []
    y = []
    data = load_data_by_label(label)
    x.extend(list(data))
    class_as_array = [0] * 12
    class_as_array[CLASSES.index(label)] = 1
    y.extend([class_as_array] * len(data))
    return shuffle(np.array(x), np.array(y))


def load_by_classes(classes: list):
    x = []
    y = []
    for cl in classes:
        data = load_data_by_label(cl)
        x.extend(list(data))
        class_as_array = [0] * 12
        class_as_array[CLASSES.index(cl)] = 1
        y.extend([class_as_array] * len(data))
    '''other_classes = load_other_classes(classes)
    x.extend(other_classes)
    class_as_array = [0] * 12
    y.extend([class_as_array] * len(other_classes))'''
    return shuffle(np.array(x), np.array(y))


if __name__ == "__main__":
    classes = ['triangle']
    load_by_classes(classes)
