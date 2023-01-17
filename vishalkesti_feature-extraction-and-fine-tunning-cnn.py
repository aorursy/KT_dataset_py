!nvidia-smi
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

 

#configure

# sets matplotlib to inline and displays graphs below the corressponding cell.

% matplotlib inline  

style.use('fivethirtyeight')

sns.set(style='whitegrid',color_codes=True)



#model selection

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier,GradientBoostingClassifier

from sklearn import metrics



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

from keras.applications.inception_v3 import InceptionV3

from keras.applications.resnet50 import ResNet50
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
base_model=VGG16(include_top=False, weights='imagenet',input_shape=(128,128,3))
print (base_model.summary())
#Feature extraction using VGG16 as baseline model

featuresTrain= base_model.predict(trainX)



#reshape to flatten feature for Train data

featuresTrain= featuresTrain.reshape(featuresTrain.shape[0], -1)



featuresVal= base_model.predict(testX)

#reshape to flatten feature for Test data

featuresVal= featuresVal.reshape(featuresVal.shape[0], -1)


model = RandomForestClassifier(400,verbose=1)

model.fit(featuresTrain, trainY)



# evaluate the model



results = model.predict(featuresVal)

print (metrics.accuracy_score(results, testY))
#Bagging classifier

model23 = BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, 

                            bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False,

                            n_jobs=None, random_state=None, verbose=1)

model23.fit(featuresTrain, trainY)



# evaluate the model



results23 = model23.predict(featuresVal)

print (metrics.accuracy_score(results23, testY))
# model32 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, subsample=1.0,

#                                      min_samples_split=2, min_samples_leaf=1, 

#                                      min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, 

#                                      min_impurity_split=None, init=None, random_state=None, max_features=None, 

#                                      verbose=0, max_leaf_nodes=None, warm_start=False,

#                                      validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)

# model32.fit(featuresTrain, trainY)



# # evaluate the model



# results32 = model32.predict(featuresVal)

# print (metrics.accuracy_score(results32, testY))
model1 = LogisticRegression(random_state=0, solver='lbfgs', dual= False, multi_class='multinomial',verbose=1, max_iter=1000).fit(featuresTrain, trainY)



model1.predict(featuresTrain)



#evaluate the model



results_1 = model1.predict(featuresVal)

print(metrics.accuracy_score(results_1,testY))
base_model_1 = InceptionV3(include_top=False, weights='imagenet',input_shape=(128,128,3))
#Feature extraction using InceptionV3 as baseline model

featuresTrain_IncepV3= base_model.predict(trainX)



#reshape to flatten feature for Train data

featuresTrain_IncepV3= featuresTrain_IncepV3.reshape(featuresTrain_IncepV3.shape[0], -1)



featuresVal_IncepV3= base_model.predict(testX)

#reshape to flatten feature for Test data

featuresVal_IncepV3= featuresVal_IncepV3.reshape(featuresVal_IncepV3.shape[0], -1)
#Random forest classifier

model2 = RandomForestClassifier(5000,verbose = 1)

model2.fit(featuresTrain_IncepV3, trainY)



# evaluate the model



results_2 = model2.predict(featuresVal_IncepV3)

print (metrics.accuracy_score(results_2, testY))
#Bagging classifier

model44 = BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, 

                            bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False,

                            n_jobs=None, random_state=None, verbose=1)

model44.fit(featuresTrain_IncepV3, trainY)



# evaluate the model



results44 = model44.predict(featuresVal_IncepV3)

print (metrics.accuracy_score(results44, testY))
#Logistic regression

model3 = LogisticRegression(random_state=0, solver='lbfgs',dual= False,max_iter=1000, multi_class='multinomial',verbose =1).fit(featuresTrain_IncepV3, trainY)



model3.predict(featuresTrain_IncepV3)



#evaluate the model



results_3 = model3.predict(featuresVal_IncepV3)

print(metrics.accuracy_score(results_3,testY))
initialModel = tf.keras.applications.VGG19(weights = 'imagenet',include_top = False, input_shape =(128,128,3))



newModel = tf.keras.Model(inputs = initialModel.input, outputs = initialModel.get_layer('block5_conv2').output)
print(newModel.summary())
featuresTrain_1= newModel.predict(trainX)



#reshape to flatten feature for Train data

featuresTrain_1= featuresTrain_1.reshape(featuresTrain_1.shape[0], -1)



featuresVal_1= newModel.predict(testX)

#reshape to flatten feature for Test data

featuresVal_1= featuresVal_1.reshape(featuresVal_1.shape[0], -1)


model = RandomForestClassifier(200)

model.fit(featuresTrain, trainY)



# evaluate the model



results = model.predict(featuresVal)

print (metrics.accuracy_score(results, testY))
model1 = LogisticRegression(random_state=0, solver='lbfgs',dual= False,max_iter=1000, multi_class='multinomial').fit(featuresTrain, trainY)



model1.predict(featuresTrain)



#evaluate the model



results_1 = model1.predict(featuresVal)

print(metrics.accuracy_score(results_1,testY))
NUM_EPOCHS = 50
keep_prob = 0.5

# Load the ImageNet VGG model. Notice we exclude the densely #connected layer at the top

vggModel= tf.keras.applications.VGG16( weights='imagenet', include_top=False, input_shape=(128, 128, 3))



vggModel.trainable= False



model = tf.keras.models.Sequential()

#We now add the vggModel directly to our new model

model.add(vggModel)

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(256, activation='relu'))

model.add(tf.keras.layers.Dropout(rate = 1 - keep_prob))

model.add(tf.keras.layers.Dense(17, activation='softmax'))



print (model.summary())
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.01),metrics=['accuracy'])



H = model.fit(trainX, trainY, epochs=NUM_EPOCHS, batch_size=32, validation_data=(testX, testY))
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

#We set the trainable parameter to True

vggModel.trainable = True



#A flag variable used to change the status. 

trainableFlag = False



for layer in vggModel.layers:

    #As commented previously we are unfreezing the below mentioned layer from the baseline model 

    #for updating the weights

    if layer.name == 'block5_conv2':

        trainableFlag = True

    layer.trainable = trainableFlag

    

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.001),metrics=['accuracy'])

H1 = model.fit(trainX, trainY, epochs=NUM_EPOCHS, batch_size=32, validation_data=(testX, testY))
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
vggModel.trainable = True

trainableFlag = False



for layer in vggModel.layers:

    if layer.name == 'block4_conv1':

        trainableFlag = True

    if layer.name == 'block4_conv2':

        trainableFlag = True

    if layer.name == 'block4_conv3':

        trainableFlag = True

    if layer.name == 'block4_pool':

        trainableFlag = True

#    if layer.name == 'block5_conv1':

#        trainableFlag = True

#     if layer.name == 'block5_conv2':

#         trainableFlag = True

#     if layer.name == 'block5_conv3':

#         trainableFlag = True

#     if layer.name == 'block5_pool':

#         trainableFlag = True

    layer.trainable = trainableFlag

    

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.0001),metrics=['accuracy'])

#model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0),metrics=['accuracy'])

#model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, 

                               # epsilon=None, decay=0.0, amsgrad=Fals,metrics=['accuracy'])

print (model.summary())

H2 = model.fit(trainX, trainY, epochs=NUM_EPOCHS, batch_size=32, validation_data=(testX, testY))
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