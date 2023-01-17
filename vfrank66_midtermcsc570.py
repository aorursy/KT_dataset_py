# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model 
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Input 
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from keras.optimizers import SGD
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from keras.datasets import fashion_mnist
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
# Any results you write to the current directory are saved as output.
def load_data(channels=0):
    # The data, shuffled and split between train and test sets:
    (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
    
    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')
    train_X /= 255
    test_X /= 255
    
    # # if using ImageDataGenerator a channel is required, default is last element
    train_X = train_X.reshape(train_X.shape[0], 28,28,1)
    test_X = test_X.reshape(test_X.shape[0], 28,28,1)

    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    return {"train_X": train_X, "train_y": train_y,
            "val_X": test_X[:5000, :], "val_y": test_y[:5000, :],
            "test_X": test_X[5000:, :], "test_y": test_y[5000:, :]}

def build_network(keep_prob=0.2, optimizer='adam'):
    inputs = Input(shape=(28,28,1), name="input")

    # convolutional block 1
    conv1 = Conv2D(64, kernel_size=(3,3), activation="relu", name="conv_1")(inputs)
    batch1 = BatchNormalization(name="batch_norm_1")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name="pool_1")(batch1)

    # convolutional block 2
    conv2 = Conv2D(32, kernel_size=(3,3), activation="relu", name="conv_2")(pool1)
    batch2 = BatchNormalization(name="batch_norm_2")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name="pool_2")(batch2)

     # convolutional block 3
    #conv3 = Conv2D(32, kernel_size=(3,3), activation="relu", name="conv_3")(pool2)
    #batch3 = BatchNormalization(name="batch_norm_3")(conv3)
    #pool3 = MaxPooling2D(pool_size=(2, 2), name="pool_3")(batch3)

    # fully connected layers
    flatten = Flatten()(pool2)
    fc1 = Dense(512, activation="relu", name="fc1")(flatten)
    d1 = Dropout(rate=keep_prob, name="dropout1")(fc1)
    fc2 = Dense(256, activation="relu", name="fc2")(fc1)
    d2 = Dropout(rate=keep_prob, name="dropout2")(fc2)

    # output layer
    output = Dense(10, activation="softmax", name="softmax")(fc2)

    # finalize and compile
    model = Model(inputs=inputs, outputs=output)    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])
    return model
def create_callbacks(name):
    tensorboard_callback = TensorBoard(log_dir= "tensorboard_log" + name, write_graph=True, write_grads=False)
    checkpoint_callback = ModelCheckpoint(filepath="model-weights-" + name + ".{epoch:02d}-{val_loss:.6f}.hdf5", monitor='val_loss',
                                          verbose=0, save_best_only=True)
    return [tensorboard_callback]
def print_model_metrics(model, data):
    loss, accuracy = model.evaluate(x=data["test_X"], y=data["test_y"])
    print("\n model test loss is "+str(loss)+" accuracy is "+str(accuracy))

    y_softmax = model.predict(data["test_X"])  # this is an n x class matrix of probabilities
    y_hat = y_softmax.argmax(axis=-1)  # this will be the class number.
    test_y = data["test_y"].argmax(axis=-1)  # our test data is also categorical
    print(classification_report(test_y, y_hat))
def create_datagen(train_X, val_X):
    train_generator = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.02,
        height_shift_range=0.02,
        horizontal_flip=True)
    train_generator.fit(train_X)
    
    val_generator = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.02,
        height_shift_range=0.02,
        horizontal_flip=True)
    val_generator.fit(val_X)
    
    return train_generator, val_generator
def fit_model(model, train_generator, val_generator, batch_size, epochs, name):
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.n // batch_size,
        callbacks=create_callbacks(name=name),
        verbose=1)
    return model
def eval_model(model, val_generator, batch_size):
    scores = model.evaluate_generator(val_generator, steps=val_generator.n // batch_size)
    print("Loss: " + str(scores[0]) + " Accuracy: " + str(scores[1]))
IMG_HEIGHT = 28
IMG_WIDTH = 28
CHANNELS = 1  # RGB
data = load_data(CHANNELS)
train_generator, val_generator = create_datagen(data["train_X"], data["val_X"])
callbacks = create_callbacks(name="midterm_run2")
model = build_network()
print(model.summary())
model.fit_generator(train_generator.flow(data["train_X"], data["train_y"], batch_size=32),
                        steps_per_epoch=len(data["train_X"]) // 32,
                        epochs=200,
                        validation_data=(data["val_X"], data["val_y"]),
                        verbose=1,
                        #callbacks=callbacks
                   )
model.save("midterm_run2.h5")
eval_model(model, val_generator, 32)
print_model_metrics(model, data)