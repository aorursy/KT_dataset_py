import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input, BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model
from keras.models import load_model
from keras.datasets import cifar10

from keras.metrics import AUC



sns.set(style='white', context='notebook', palette='deep')
def get_mnist():

    train = pd.read_csv("/kaggle/input/mnist-in-csv/mnist_train.csv")
    test = pd.read_csv("/kaggle/input/mnist-in-csv/mnist_test.csv")

    Y_train = train["label"]
    Y_test = test["label"]


    # Drop 'label' column
    X_train = train.drop(labels = ["label"],axis = 1) 
    X_test = test.drop(labels = ["label"],axis = 1) 


    # free some space
    train = None
    test = None

    X_train = X_train / 255.0
    X_test = X_test / 255.0



    X_train = X_train.values.reshape(-1,28,28,1)
    X_test = X_test.values.reshape(-1,28,28,1)


    Y_train = to_categorical(Y_train, num_classes = 10)
    Y_test = to_categorical(Y_test, num_classes = 10)

    # Set the random seed
    random_seed = 2

    
    return X_train , X_test, Y_train , Y_test


def get_cifar10():
    (x_train,y_train),(x_test,y_test) = cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    y_train = to_categorical(y_train, num_classes = 10)
    y_test = to_categorical(y_test, num_classes = 10)
    
    return x_train, x_test, y_train, y_test


def get_fashion_mnist():
    train = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")
    test = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_test.csv")

    Y_train = train["label"]
    Y_test = test["label"]


    # Drop 'label' column
    X_train = train.drop(labels = ["label"],axis = 1) 
    X_test = test.drop(labels = ["label"],axis = 1) 


    # free some space
    train = None
    test = None

    X_train = X_train / 255.0
    X_test = X_test / 255.0


    X_train = X_train.values.reshape(-1,28,28,1)
    X_test = X_test.values.reshape(-1,28,28,1)


    Y_train = to_categorical(Y_train, num_classes = 10)
    Y_test = to_categorical(Y_test, num_classes = 10)

    # Set the random seed
    random_seed = 2

    
    return X_train, X_test, Y_train, Y_test


def get_simple_model(shape):
    # Set the CNN model 
    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                     activation ='relu', input_shape = shape))
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                     activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))


    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = "softmax"))
    
    # Compile the model
    model.compile(optimizer = "adam" , loss = "categorical_crossentropy", metrics=["accuracy",AUC()])
    
    return model

def get_vgg16_model(shape):
    
    model = Sequential()

    model.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=shape))
    model.add(Conv2D(64, 3, activation='relu', padding='same'))
    model.add(MaxPool2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(128, 3, activation='relu', padding='same'))
    model.add(Conv2D(128, 3, activation='relu', padding='same'))
    model.add(MaxPool2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(256, 3, activation='relu', padding='same'))
    model.add(Conv2D(256, 3, activation='relu', padding='same'))
    model.add(Conv2D(256, 3, activation='relu', padding='same'))
    model.add(MaxPool2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(512, 3, activation='relu', padding='same'))
    model.add(Conv2D(512, 3, activation='relu', padding='same'))
    model.add(Conv2D(512, 3, activation='relu', padding='same'))
    model.add(MaxPool2D(2, 1)) # default stride is 2
    model.add(BatchNormalization())

    model.add(Conv2D(512, 3, activation='relu', padding='same'))
    model.add(Conv2D(512, 3, activation='relu', padding='same'))
    model.add(Conv2D(512, 3, activation='relu', padding='same'))
    model.add(MaxPool2D(2, 1)) # default stride is 2
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer = "adam" , loss = "categorical_crossentropy", metrics=["accuracy",AUC()])
    
    return model
datasets = ["mnist","cifar10","fashion_mnist"]
models = ["simple", "vgg16"]
for dataset in datasets:
    X_train, X_test, Y_train, Y_test = eval(f'get_{dataset}()')
    for model_name in models:
        print(dataset + " "+ model_name)
        model = eval(f'get_{model_name}_model({X_train.shape[1:]})')
        history = model.fit(X_train,Y_train, batch_size=128,
                                      epochs = 10, validation_data = (X_test, Y_test),
                                      verbose = 1)
        model.save(f'./{model_name}_{dataset}.h5')
def transfer(archi, dataset, stoping_layer):
    model = load_model(f"/kaggle/input/transfer-models/{archi}_{dataset}.h5")
    model = Model(model.input, model.layers[-stoping_layer].output)
    for layer in model.layers:
        layer.trainable = False
        
    # add the FC part
    x = Flatten()(model.output)
    x = Dense(50,activation="relu")(x)
    x = Dense(10,activation="softmax")(x)
    model = Model(model.input, x)
    model.compile(optimizer = "adam" , loss = "categorical_crossentropy",  metrics=["accuracy",AUC()])
    return model

def plot_history(history, history2, stop, archi):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history2.history['accuracy'])
    plt.plot(history2.history['val_accuracy'])
    plt.title(f'accuracy {archi} - stoping at layer {stop} from the bottom')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_mnist', 'test_mnist', 'train_cifar10', 'test_cifar10'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(history2.history['loss'])
    plt.plot(history2.history['val_loss'])
    plt.title(f'{archi} loss trained on {dataset} - stoping at layer {stop} from the bottom')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_mnist', 'test_mnist', 'train_cifar10', 'test_cifar10'], loc='upper left')
    plt.show()
    print()
    
def pad_dataset(train, test):
    train = np.pad(X_train, [(0, 0),(0, 4),(0, 4),(0, 0)], mode='constant', constant_values=0)
    test = np.pad(test, [(0, 0),(0, 4),(0, 4),(0, 0)], mode='constant', constant_values=0)
    train = np.stack((train,) * 3, axis=-1).reshape(train.shape[0], train.shape[1], train.shape[2], 3)
    test = np.stack((test,) * 3, axis=-1).reshape(test.shape[0], test.shape[1], test.shape[2], 3)
    return train,test
vgg_stoping_layers = [7, 12, 17, 22, 26]
simple_stoping_layers = [6, 10]
archis = ["vgg16", "simple"]
datasets = ["mnist","cifar10"]
X_train, X_test, Y_train, Y_test = get_fashion_mnist()
np.save(f'./y_test.npy', Y_test)
for archi in archis:
    stoping_points = vgg_stoping_layers if archi == "vgg16" else simple_stoping_layers
    for stop in stoping_points:
        X_train, X_test, Y_train, Y_test = get_fashion_mnist()
        histories = []
        for dataset in datasets:
            if dataset == "cifar10":
                X_train, X_test = pad_dataset(X_train, X_test)
            transfer_model = transfer(archi,dataset,stop)
            print(dataset + " "+ archi)
            history = transfer_model.fit(X_train,Y_train, batch_size=128,
                                              epochs = 10, validation_data = (X_test, Y_test),
                                              verbose = 0)
            histories.append(history)
            preds = transfer_model.predict(X_test)
            np.save(f'./{dataset}_{archi}_{stop}.npy', preds)
        plot_history(histories[0],histories[1], stop , archi)
