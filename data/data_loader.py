from keras.utils import image_dataset_from_directory

all_levels = ['easy', 'medium_easy', 'medium_hard', 'hard']
images_levels = ['easy']
base_url = '../generated_images/dataset'
IMG_SIZE = 96
BATCH_SIZE = 100


def load_dataset():
    train, val = None, None
    for level in images_levels:
        train_ds = image_dataset_from_directory(
            validation_split=0.2, subset="training", seed=123,
            directory=base_url + '/' + level, labels='inferred', batch_size=BATCH_SIZE,
            image_size=(IMG_SIZE, IMG_SIZE))
        val_ds = image_dataset_from_directory(
            validation_split=0.2, subset="validation", seed=123,
            directory=base_url + '/' + level, labels='inferred', batch_size=BATCH_SIZE,
            image_size=(IMG_SIZE, IMG_SIZE))
        if train is None and val is None:
            train, val = train_ds, val_ds
        else:
            train = train.concatenate(train_ds)
            val = val.concatenate(val_ds)
    return train, val
