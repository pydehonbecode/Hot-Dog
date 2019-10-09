import pandas as pd
import os
from sklearn.model_selection import train_test_split
import pickle
from keras.preprocessing.image import ImageDataGenerator
from .model import create_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
print("real", os.path.realpath(__file__))

TRAINING_DATA_FILE = os.path.dirname(os.path.realpath(__file__)) + '/dataset/train'
TESTING_DATA_FILE = os.path.dirname(os.path.realpath(__file__)) + '/dataset/test'
MODEL_PATH = os.path.dirname(os.path.realpath(__file__)) + '/saved_model/weigths.h5'
ARCHITECTURE_PATH = os.path.dirname(os.path.realpath(__file__)) + '/saved_model/architecture.json'
MODEL_PATH = os.path.dirname(os.path.realpath(__file__)) + '/saved_model/best_model.h5'


def _save_model(model):
    pickle.dump(model, open(MODEL_PATH, 'wb'))


def train_model():
    K.clear_session()
    batch_size = 16
    model = create_model()
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=45,
        vertical_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        TRAINING_DATA_FILE,  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        TESTING_DATA_FILE,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

    mc = ModelCheckpoint(MODEL_PATH, monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    model.fit_generator(
        train_generator,
        callbacks=[mc],
        steps_per_epoch=2000 // batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
    """model.save_weights(MODEL_PATH)
    # Save the model architecture
    with open(ARCHITECTURE_PATH, 'w') as f:
        f.write(model.to_json())"""
    K.clear_session()
    return model.history


if __name__ == '__main__':
    print(train_model())
