from keras.utils import image_dataset_from_directory

train_url = '../../generated_images/dataset'
test_url = '../../generated_images_test/dataset'
IMG_SIZE = 96
BATCH_SIZE = 100


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
