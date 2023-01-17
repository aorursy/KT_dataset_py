# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#This is helpful in visualising matplotlib graphs
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow import keras #The deep learning model we will use to train our dataset will make use of this
import tensorflow as tf
from PIL import Image as IMG #To read the image file
import os #To move through the folders and fetching the images
import matplotlib.pyplot as plt #To render Plots of our data
import sklearn.model_selection as smodel #To split the data for training and cross validation set
import skimage.data as dt
from skimage import transform 
from skimage.color import rgb2gray
import random
import keras
from random import shuffle
from keras.models import Sequential #This is to create sequential model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten #this is for creating different layers
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.imagenet_utils import decode_predictions
IMG_SIZE = 200
EPOCHS = 100
NO_OF_IMG = 3000
NUM_CLASSES = 2
# Any results you write to the current directory are saved as output.
# Any results you write to the current directory are saved as output.
print(os.listdir("../input/"))
!ls "../input/full-keras-pretrained-no-top"
def seperateCatDog(root_dir):
    dog = []
    cat = []
    imagefiles = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]
    for image in imagefiles:
        category = image.split('/')[-1].split('.')[-3]
        if(category=="cat"):
            cat.append(image)
        else:
            dog.append(image)
    return cat[:NO_OF_IMG]+dog[:NO_OF_IMG]
def making_matrices(root_dir):
    count = 0
    imagefiles = seperateCatDog(root_dir)
    images = np.zeros((len(imagefiles),IMG_SIZE,IMG_SIZE,3))
    labels = np.zeros((len(imagefiles)),dtype='S140')
    for image in imagefiles:
        images[count] = img_to_array(load_img(image, target_size=(IMG_SIZE, IMG_SIZE)))
        labels[count] = image.split('/')[-1].split('.')[-3]
        count += 1
    return images,labels
rootdir = '../input/dogs-vs-cats/train/train/'
images,label = making_matrices(rootdir)
print("Images Shape",images.shape)
print("Label Shape",label.shape)
#Let's Visualize some images 
plt.figure(figsize=(10,10))
rand = np.random.randint(0,NO_OF_IMG*2,25)
for i in range(len(rand)):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(images[rand[i]])
    plt.xlabel(label[rand[i]])
"""Let's check for the bias of image"""
plt.figure()
plt.hist(label)
plt.savefig("labelbias.png")
"""Since both label are equal hence there is no bias in label
But the label are string let's change them to integer to do that let's make a dictionary with string label as key and a int as value """
labeldict = {}
for i in range(len(np.unique(label)[:])):
    labeldict[np.unique(label)[i]] = i

def yvectorize(dict,data):
    '''This will assign the numeric label to each string label in the label matrix'''
    return dict[data]
vect = np.vectorize(yvectorize)
label = vect(labeldict,label)


def maporiginallabel(dic,data):
    """This will reverse map the label i.e given a int label it will 
    return the original string label"""
    for key, value in dic.items():    # for name, age in list.items():  (for Python 3.x)
        if(value == data):
            return (key)
one_hot_labels = to_categorical(label,NUM_CLASSES)
label = None
print("New Label Shape",one_hot_labels.shape)
#Lets's split training data to trainig set and cross validation set
x_train,x_cross,y_train,y_cross = smodel.train_test_split(images,one_hot_labels,test_size=0.3)
images = None
one_hot_labels = None
modelweight= "../input/full-keras-pretrained-no-top/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
base_model = ResNet50(
    weights = modelweight, include_top=False, input_shape=(IMG_SIZE,IMG_SIZE, 3))

# Freeze the layers except the last 4 layers
for layer in base_model.layers[:]:
    layer.trainable = False
 
# Check the trainable status of the individual layers
for layer in base_model.layers:
    print(layer, layer.trainable)
base_model.summary()
# Create the model
model = Sequential()
 
# Add the vgg convolutional base model
model.add(base_model)
 
# Add new layers
# flatten
model.add(Flatten())

model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
 
# Show a summary of the model. Check the number of trainable parameters
model.summary()
x_train.shape,x_cross.shape
"""Let's increase the no. of images and fit the model"""
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
history = model.fit(x_train,y_train,epochs=EPOCHS,
                   validation_data=(x_cross,y_cross),
                   verbose = 1)
"""Let's test our model with more generated data"""
cross_loss, cross_acc = model.evaluate(x_cross,y_cross)
train_loss, train_acc = model.evaluate(x_train,y_train)
predictions = np.argmax(model.predict(x_train),axis=1)
print("Train Accuracy",train_acc)
print("Cross Validation Accuracy",cross_acc)
# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.savefig("AccuracyErrorCurveResnet.png")
model.save('Team_Model_ResNet.h5')  # creates a HDF5 file 'Team_Model_ResNet.h5'
del model  # deletes the existing model