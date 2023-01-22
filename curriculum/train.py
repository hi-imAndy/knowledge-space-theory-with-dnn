import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from data.data_loader import load_dataset, load_by_classes, load_by_class
from dnn.model import setup_hard_model

LEARNING_ORDER = [8, 4, 0, 9, 5, 1, 12, 7, 3, 11, 6, 2]
LEARNING_CLASSES = ['triangle', 'square', 'ellipse', 'triangle_curved', 'square_curved', 'ellipse_curved',
                    'triangle_line', 'square_line', 'ellipse_line', 'triangle_hard', 'square_hard', 'ellipse_line']


def plot_history(history, iteration):
    plt.plot(history.history['val_accuracy'], label=f'val_accuracy {iteration}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')


def train():
    model = setup_hard_model()
    train_ds, val_ds = load_dataset()
    history = model.fit(train_ds, validation_data=val_ds, epochs=30)
    plot_history(history)
    predictions = model.predict(val_ds)
    print(predictions)


def get_train_test_by_label(label):
    x, y = load_by_class(label)
    x = x.reshape(len(x), 96, 96, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


def train_curriculum():
    model = setup_hard_model()
    X_train, X_test, Y_train, Y_test = None, None, None, None
    for i in range(0, 3):
        label = LEARNING_CLASSES[i]
        if i == 0:
            X_train, X_test, Y_train, Y_test = get_train_test_by_label(label)
        else:
            x_train, x_test, y_train, y_test = get_train_test_by_label(label)
            X_train = np.concatenate((X_train, x_train))
            Y_train = np.concatenate((Y_train, y_train))
            X_test = np.concatenate((X_test, x_test))
            Y_test = np.concatenate((Y_test, y_test))
    X_train, Y_train = shuffle(X_train, Y_train)
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10)
    plot_history(history, 1)
    for i in range(3, len(LEARNING_CLASSES)):
        label = LEARNING_CLASSES[i]
        x_train, x_test, y_train, y_test = get_train_test_by_label(label)
        X_train = np.concatenate((X_train, x_train))
        Y_train = np.concatenate((Y_train, y_train))
        X_test = np.concatenate((X_test, x_test))
        Y_test = np.concatenate((Y_test, y_test))
    X_train, Y_train = shuffle(X_train, Y_train)
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=20)
    plot_history(history, 2)
    plt.savefig('../../plots/curriculum_accuracy.png')


if __name__ == "__main__":
    # train()
    train_curriculum()

