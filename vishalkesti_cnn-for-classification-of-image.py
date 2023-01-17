# Ignore  the warnings

import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



# data visualisation and manipulation

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

import scipy

 

#configure

# sets matplotlib to inline and displays graphs below the corressponding cell.

% matplotlib inline  

style.use('fivethirtyeight')

sns.set(style='whitegrid',color_codes=True)



#model selection

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from scipy import stats

#preprocess.

from keras.preprocessing.image import ImageDataGenerator



#dl libraraies

from keras import backend as K

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from keras.utils import to_categorical

from keras.callbacks import ReduceLROnPlateau



# specifically for cnn

#from conv.conv import ShallowNet

from keras.layers import Dropout, Flatten,Activation

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

 

import tensorflow as tf

import random as rn



# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.

import cv2 

import h5py

import numpy as np  

from tqdm import tqdm

import os                   

from random import shuffle  

from zipfile import ZipFile

from PIL import Image



#TL pecific modules

from keras.applications.vgg16 import VGG16
# from google.colab import drive

# drive.mount('/content/gdrive')
#tf.reset_default_graph()
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def loadDataH5():

    with h5py.File('/kaggle/input/data1h5/data1.h5','r') as hf:

        trainX = np.array(hf.get('trainX'))

        trainY = np.array(hf.get('trainY'))

        valX = np.array(hf.get('valX'))

        valY = np.array(hf.get('valY'))

        print (trainX.shape,trainY.shape)

        print (valX.shape,valY.shape)

    return trainX, trainY, valX, valY

trainX, trainY, testX, testY = loadDataH5()
type(trainX)
# flower17 class names

class_names = ["daffodil", "snowdrop", "lilyvalley", "bluebell", "crocus",

			   "iris", "tigerlily", "tulip", "fritillary", "sunflower", 

			   "daisy", "coltsfoot", "dandelion", "cowslip", "buttercup",

			   "windflower", "pansy"]
# np.random.seed(42)

# rn.seed(42)

# tf.set_random_seed(42)
#We are going to initialize batch size and the number of epochs which is going

# to be used across the code

batch_size = 32

epochs=50
#tf.reset_default_graph()
def singleCNN(width, height, depth, classes):

    # initialize the model along with the input shape to be "channels last"

    model = tf.keras.Sequential() 

    inputShape = (height, width, depth)



    # define the first (and only) CONV => RELU layer

    model.add(tf.keras.layers.Conv2D (64, (3, 3), padding="same", input_shape=inputShape, activation='relu'))

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



    # softmax classifier

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(classes, activation='softmax'))

  

    return model
# initialize the optimizer and model

print("Compiling model...")



opt = tf.keras.optimizers.SGD(lr=0.01)

model = singleCNN(width=128, height=128, depth=3, classes=17)

print (model.summary())



model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,

	metrics=["accuracy"])



# train the network

print("Training network...")

H = model.fit(trainX, trainY, validation_data=(testX, testY),	batch_size=32, epochs=epochs)
# plot the training loss and accuracy

plt.style.use("ggplot")

plt.figure()

plt.plot(np.arange(0, 50), H.history["loss"], label="train_loss")

plt.plot(np.arange(0, 50), H.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, 50), H.history["acc"], label="train_acc")

plt.plot(np.arange(0, 50), H.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()

plt.show()

# #Declare this as global:

# global graph

# graph = tf.get_default_graph()

#tf.reset_default_graph()
def firstCNNVariant(width, height, depth, classes):

  

    # initialize the model along with the input shape

    model1 = tf.keras.Sequential()

    inputShape = (height, width, depth)

    # first set of CONV => RELU => POOL layers

    model1.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=inputShape, activation='relu'))

    model1.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL layers

    model1.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'))

    model1.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # first (and only) set of FC => RELU layers

    model1.add(tf.keras.layers.Flatten())

    model1.add(tf.keras.layers.Dense(500, activation='relu'))

    # softmaxclassifier

    model1.add(tf.keras.layers.Dense(classes, activation='softmax'))

     

    return model1
from keras import backend as K

# initialize the optimizer and model

print("Compiling model...")

# with graph.as_default():

opt = tf.keras.optimizers.SGD(lr=0.01)

#opt = SGD(lr = 0.01)

model1 = firstCNNVariant(width=128, height=128, depth=3, classes=17)

print (model1.summary())



model1.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])



# train the network

print("Training network...")

H1 = model1.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=epochs, verbose=1)
# plot the training loss and accuracy

plt.style.use("ggplot")

plt.figure()

plt.plot(np.arange(0, 50), H1.history["loss"], label="train_loss")

plt.plot(np.arange(0, 50), H1.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, 50), H1.history["acc"], label="train_acc")

plt.plot(np.arange(0, 50), H1.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()

plt.show()
#tf.reset_default_graph()
def SecondCNNVariant(width, height, depth, classes):

    # initialize the model along with the input shape

    model2 = tf.keras.Sequential()

    inputShape = (height, width, depth)

    # first CONV => CONV => POOL layer set

    model2.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=inputShape, activation='relu'))

    model2.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same",activation='relu'))

    model2.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second CONV => CONV => POOL layer set

    model2.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same",activation='relu'))

    model2.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same",activation='relu'))

    model2.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # first (and only) set of FC => RELU layers

    model2.add(tf.keras.layers.Flatten())

    model2.add(tf.keras.layers.Dense(512,activation='relu'))

    # softmaxclassifier

    model2.add(tf.keras.layers.Dense(classes, activation='softmax'))



    return model2
# initialize the optimizer and model



print("Compiling model...")



opt = tf.keras.optimizers.SGD(lr=0.01)

#opt = SGD(lr = 0.01)

model2 = SecondCNNVariant(width=128, height=128, depth=3, classes=17)

print (model2.summary())



model2.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])



# train the network

print("Training network...")

H2 = model2.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=epochs, verbose=1)
# plot the training loss and accuracy

plt.style.use("ggplot")

plt.figure()

plt.plot(np.arange(0, 50), H2.history["loss"], label="train_loss")

plt.plot(np.arange(0, 50), H2.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, 50), H2.history["acc"], label="train_acc")

plt.plot(np.arange(0, 50), H2.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()

plt.show()
def thirdCNNVariant(width, height, depth, classes):

    # initialize the model along with the input shape

    model45 = tf.keras.Sequential()

    inputShape = (height, width, depth)

    # first set of CONV => RELU => POOL layers

    model45.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=inputShape, activation='relu'))

    model45.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL layers

    model45.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'))

    model45.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # third set of CONV => RELU => POOL layers

    model45.add(tf.keras.layers.Conv2D(96, (3, 3), padding="same", activation='relu'))

    model45.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # first (and only) set of FC => RELU layers

    model45.add(tf.keras.layers.Flatten())

    model45.add(tf.keras.layers.Dense(500, activation='relu'))

    # softmaxclassifier

    model45.add(tf.keras.layers.Dense(classes, activation='softmax'))

    return model45

# le=LabelEncoder()

# Y=le.fit_transform(trainY)

# Y=to_categorical(Y,17)



#Construct the image generator for data augmentation

datagen1 = ImageDataGenerator(

        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

        )



datagen1.fit(trainX)

datagen1.fit(testX)
#Construct the image generator for data augmentation



datagen2 = ImageDataGenerator(

        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.2, # Randomly zoom image 

        horizontal_flip=True,  # randomly flip images

        )





datagen2.fit(trainX)

datagen2.fit(testX)
#Construct the image generator for data augmentation



datagen3 = ImageDataGenerator(

        shear_range = 0.2,

        zoom_range = 0.2,

        rotation_range = 30,

        horizontal_flip = True,

        vertical_flip = True

        )  





datagen3.fit(trainX)

datagen3.fit(testX)
#Construct the image generator for data augmentation



datagen4 = ImageDataGenerator(

        

        rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.4, # Randomly zoom image 

        horizontal_flip = False,

        vertical_flip=True)  # randomly flip images





datagen4.fit(trainX)

datagen4.fit(testX)
# initialize the optimizer and model



print("Compiling model...")



opt = tf.keras.optimizers.SGD(lr=0.01)

#opt = SGD(lr = 0.01)

model = singleCNN(width=128, height=128, depth=3, classes=17)

print (model.summary())



model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])



# train the network

print("Training network...")

H3 = model.fit_generator(datagen1.flow(trainX,trainY, batch_size=batch_size),

                              epochs = epochs, validation_data = datagen1.flow(testX,testY, batch_size=batch_size),

                              verbose = 1, steps_per_epoch=trainX.shape[0] // batch_size, validation_steps = testX.shape[0] //batch_size)
# plot the training loss and accuracy

plt.style.use("ggplot")

plt.figure()

plt.plot(np.arange(0, 50), H3.history["loss"], label="train_loss")

plt.plot(np.arange(0, 50), H3.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, 50), H3.history["acc"], label="train_acc")

plt.plot(np.arange(0, 50), H3.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()

plt.show()
# initialize the optimizer and model



print("Compiling model...")



opt = tf.keras.optimizers.SGD(lr=0.01)

#opt = SGD(lr = 0.01)

model = singleCNN(width=128, height=128, depth=3, classes=17)

print (model.summary())



model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])



# train the network

print("Training network...")

H4 = model.fit_generator(datagen2.flow(trainX,trainY, batch_size=batch_size),

                              epochs = epochs, validation_data = datagen2.flow(testX,testY, batch_size=batch_size),

                              verbose = 1, steps_per_epoch=trainX.shape[0] // batch_size, validation_steps = testX.shape[0] //batch_size)
# plot the training loss and accuracy

plt.style.use("ggplot")

plt.figure()

plt.plot(np.arange(0, 50), H4.history["loss"], label="train_loss")

plt.plot(np.arange(0, 50), H4.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, 50), H4.history["acc"], label="train_acc")

plt.plot(np.arange(0, 50), H4.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()

plt.show()
# initialize the optimizer and model



print("Compiling model...")



opt = tf.keras.optimizers.SGD(lr=0.01)

#opt = SGD(lr = 0.01)

model = singleCNN(width=128, height=128, depth=3, classes=17)

print (model.summary())



model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])



# train the network

print("Training network...")

H5 = model.fit_generator(datagen3.flow(trainX,trainY, batch_size=batch_size),

                              epochs = epochs, validation_data = datagen3.flow(testX,testY, batch_size=batch_size),

                              verbose = 1, steps_per_epoch=trainX.shape[0] // batch_size, validation_steps = testX.shape[0] //batch_size)
# plot the training loss and accuracy

plt.style.use("ggplot")

plt.figure()

plt.plot(np.arange(0, 50), H5.history["loss"], label="train_loss")

plt.plot(np.arange(0, 50), H5.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, 50), H5.history["acc"], label="train_acc")

plt.plot(np.arange(0, 50), H5.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()

plt.show()
# initialize the optimizer and model



print("Compiling model...")



opt = tf.keras.optimizers.SGD(lr=0.01)

#opt = SGD(lr = 0.01)

model = singleCNN(width=128, height=128, depth=3, classes=17)

print (model.summary())



model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])



# train the network

print("Training network...")

H6 = model.fit_generator(datagen4.flow(trainX,trainY, batch_size=batch_size),

                              epochs = epochs, validation_data = datagen4.flow(testX,testY, batch_size=batch_size),

                              verbose = 1, steps_per_epoch=trainX.shape[0] // batch_size, validation_steps = testX.shape[0] //batch_size)
# plot the training loss and accuracy

plt.style.use("ggplot")

plt.figure()

plt.plot(np.arange(0, 50), H6.history["loss"], label="train_loss")

plt.plot(np.arange(0, 50), H6.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, 50), H6.history["acc"], label="train_acc")

plt.plot(np.arange(0, 50), H6.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()

plt.show()
def fourthCNNVariant(width, height, depth, classes):

    # initialize the model along with the input shape

    model46 = tf.keras.Sequential()

    inputShape = (height, width, depth)

    # first set of CONV => RELU => POOL layers

    model46.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=inputShape, activation='relu'))

    model46.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL layers

    model46.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'))

    model46.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # third set of CONV => RELU => POOL layers

    model46.add(tf.keras.layers.Conv2D(96, (3, 3), padding="same", activation='relu'))

    model46.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # forth set of CONV => RELU => POOL layers

    model46.add(tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'))

    model46.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # first (and only) set of FC => RELU layers

    model46.add(tf.keras.layers.Flatten())

    model46.add(tf.keras.layers.Dense(500, activation='relu'))

    # softmaxclassifier

    model46.add(tf.keras.layers.Dense(classes, activation='softmax'))

    return model46
def fifthCNNVariant(width, height, depth, classes):

    # initialize the model along with the input shape

    model48 = tf.keras.Sequential()

    inputShape = (height, width, depth)

    # first set of CONV => RELU => POOL layers

    model48.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=inputShape, activation='relu'))

    model48.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL layers

    model48.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'))

    model48.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # third set of CONV => RELU => POOL layers

    model48.add(tf.keras.layers.Conv2D(96, (3, 3), padding="same", activation='relu'))

    model48.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # forth set of CONV => RELU => POOL layers

    model48.add(tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'))

    model48.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # fifth set of CONV => RELU => POOL layers

    model48.add(tf.keras.layers.Conv2D(160, (3, 3), padding="same", activation='relu'))

    model48.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # first (and only) set of FC => RELU layers

    model48.add(tf.keras.layers.Flatten())

    model48.add(tf.keras.layers.Dense(500, activation='relu'))

    # softmaxclassifier

    model48.add(tf.keras.layers.Dense(classes, activation='softmax'))

    return model48
def train_Model(trainX,trainY,testX,testY):

    print("Initializing various models")

    models = []

    models.append(singleCNN(width=128, height=128, depth=3, classes=17))

    models.append(firstCNNVariant(width=128, height=128, depth=3, classes=17))

    models.append(thirdCNNVariant(width=128, height=128, depth=3, classes=17))

    models.append(fourthCNNVariant(width=128, height=128, depth=3, classes=17))

    models.append(fifthCNNVariant(width=128, height=128, depth=3, classes=17))

    print("The total number of models", len(models))

    val = []

    for i in np.arange(0,len(models)):

        print("[INFO] training model {}/{}".format(i+1, len(models)))

        

        models[i].compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        H8 = models[i].fit_generator(datagen2.flow(trainX,trainY, batch_size=batch_size),

                              epochs = epochs, validation_data = datagen2.flow(testX,testY, batch_size=batch_size),

                              verbose = 1, steps_per_epoch=trainX.shape[0] // batch_size,)

        





        val.append(models[i])

    # plot the training loss and accuracy

        N = epochs

        p = ['model_{}.png'.format(i)]

        plt.style.use('ggplot')

        plt.figure()

        plt.plot(np.arange(0, N), H8.history['loss'],

                 label='train_loss')

        plt.plot(np.arange(0, N), H8.history['val_loss'],

                 label='val_loss')

        plt.plot(np.arange(0, N), H8.history['acc'],

                 label='train-acc')

        plt.plot(np.arange(0, N), H8.history['val_acc'],

                 label='val-acc')

        plt.title("Training Loss and Accuracy for model {}".format(i))

        plt.xlabel("Epoch #")

        plt.ylabel("Loss/Accuracy")

        plt.legend()

        #plt.savefig(os.path.sep.join(p))

        plt.close()

    return val



              
import scipy

def predict(val , testX, testY):



    # load the trained convolutional neural network

    print("[INFO] loading network...")

    models = val





    labelName = class_names





    print("[INFO] evaluating ensemble...")

    predictions = []

    accuracy_model = []

    

    for model in models:

        

        predictions.append(model.predict(testX,batch_size=64))

        

        print("##############################################################")

    print("[INFO] Ensemble with Averaging")

    

    predictions = np.average(predictions,axis=0)

      

    print("##############################################################")

    print('\n')

    print("[INFO] Ensemble with voting")

    

    labels = []

    for m in models:

        predicts = np.argmax(m.predict(testX, batch_size=64), axis=1)

        labels.append(predicts)

    #print("labels_append:", labels)

    # Ensemble with voting

    labels = np.array(labels)

    #print("labels_array:", labels)

    

    labels = np.transpose(labels, (1, 0))

    #print("labels_transpose:", labels)

        

    labels = scipy.stats.mode(labels, axis=1)[0]

    #print("labels_mode:", labels)

    labels = np.squeeze(labels)

    #print("labels: ", labels)

    print(classification_report(testY,labels, target_names=labelName))

    accu = accuracy_score(testY, labels)

    return accu

if __name__ == '__main__':

    val = train_Model(trainX,trainY,testX,testY)

    accuracy_ensemble = predict(val, testX, testY)

    print('The accuracy of the ensemble model obtained is : ', accuracy_ensemble)
def fixedLearner(trainX,trainY,testX,testY,width, height, depth, classes):

    opt = tf.keras.optimizers.SGD(lr=0.01)

    # initialize the model along with the input shape

    model = tf.keras.Sequential()

    inputShape = (height, width, depth)

    # first set of CONV => RELU => POOL layers

    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=inputShape, activation='relu'))

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL layers

    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'))

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # first (and only) set of FC => RELU layers

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(500, activation='relu'))

    # softmaxclassifier

    model.add(tf.keras.layers.Dense(classes, activation='softmax'))

    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    #fit model

    model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=epochs, verbose=1)

     

    return model
def ensembleEvaluation(ensemble_members, testX, testY):

    models = ensemble_members

    predictions = []

    labels = []

    for model in models:

        predictions.append(model.predict(testX,batch_size=64))

    predictions = np.average(predictions,axis=0)

    

    

    for m in models:

        predicts = np.argmax(m.predict(testX, batch_size=64), axis=1)

        labels.append(predicts)

    #print("labels_append:", labels)

    # Ensemble with voting

    labels = np.array(labels)

    #print("labels_array:", labels)

    

    labels = np.transpose(labels, (1, 0))

    #print("labels_transpose:", labels)

        

    labels = scipy.stats.mode(labels, axis=1)[0]

    #print("labels_mode:", labels)

    labels = np.squeeze(labels)

    #print("labels: ", labels)

    #print(classification_report(testY,labels, target_names=labelName))

    result = accuracy_score(testY, labels)

    return predictions, result
# initialize the optimizer and model

width=128

height=128

depth=3

classes=17

#Number of times we want our baselearner to run

num_of_iteration = 10



#The number of ensemble member for each run

ensemble_members = [fixedLearner(trainX,trainY,testX,testY,width, height, depth, classes) for _ in range(num_of_iteration) ]



#Individual accuracy score

accuracy_single = []

#Ensemble accuracy score

accuracy_ensemble = []

labels = []

#Calculation of accuracy for each model

  

predictions, accuracy_ensemble = ensembleEvaluation(ensemble_members, testX, testY)
#Combined accuracy of the final model

#print('Single accuracy:',accuracy_single)

#print(np.std(accuracy_single))

#print(np.std(predictions))

print('Ensemble accuracy: ',accuracy_ensemble)
# # plot score vs number of ensemble members

# x_axis = [i for i in range(1, len(ensemble_members)+1)]

# pyplot.plot(x_axis, accuracy_single, marker='o', linestyle='None')

# pyplot.plot(x_axis, accuracy_ensemble, marker='o')

# pyplot.show()