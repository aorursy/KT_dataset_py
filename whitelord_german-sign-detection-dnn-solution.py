# Libraries 

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import tensorflow as tf

import cv2

from PIL import Image

import os

# Reading the input images and putting them into a numpy array

data=[]

labels=[]



height = 30

width = 30

channels = 3

classes = 43

n_inputs = height * width*channels



for i in range(classes) :

    path = "../input/train/{0}/".format(i)

    print(path)

    Class=os.listdir(path)

    for a in Class:

        try:

            image=cv2.imread(path+a)

            image_from_array = Image.fromarray(image, 'RGB')

            size_image = image_from_array.resize((height, width))

            data.append(np.array(size_image))

            labels.append(i)

        except AttributeError:

            print(" ")

            

Cells=np.array(data)

labels=np.array(labels)



#Randomize the order of the input images

s=np.arange(Cells.shape[0])

np.random.seed(43)

np.random.shuffle(s)

Cells=Cells[s]

labels=labels[s]
#Spliting the images into train and validation sets

(X_train,X_val)=Cells[(int)(0.2*len(labels)):],Cells[:(int)(0.2*len(labels))]

X_train = X_train.astype('float32')/255 

X_val = X_val.astype('float32')/255

(y_train,y_val)=labels[(int)(0.2*len(labels)):],labels[:(int)(0.2*len(labels))]



#Using one hote encoding for the train and validation labels

from keras.utils import to_categorical

y_train = to_categorical(y_train, 43)

y_val = to_categorical(y_val, 43)
#Definition of the DNN model



from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout



model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))

model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(rate=0.5))

model.add(Dense(43, activation='softmax'))



#Compilation of the model

model.compile(

    loss='categorical_crossentropy', 

    optimizer='adam', 

    metrics=['accuracy']

)
#using ten epochs for the training and saving the accuracy for each epoch

epochs = 10

history = model.fit(X_train, y_train, batch_size=32, epochs=epochs,

validation_data=(X_val, y_val))



#Display of the accuracy and the loss values

import matplotlib.pyplot as plt



plt.figure(0)

plt.plot(history.history['acc'], label='training accuracy')

plt.plot(history.history['val_acc'], label='val accuracy')

plt.title('Accuracy')

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.legend()



plt.figure(1)

plt.plot(history.history['loss'], label='training loss')

plt.plot(history.history['val_loss'], label='val loss')

plt.title('Loss')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()
#Predicting with the test data

y_test=pd.read_csv("../input/Test.csv")

labels=y_test['Path'].as_matrix()

y_test=y_test['ClassId'].values



data=[]



for f in labels:

    image=cv2.imread('../input/test/'+f.replace('Test/', ''))

    image_from_array = Image.fromarray(image, 'RGB')

    size_image = image_from_array.resize((height, width))

    data.append(np.array(size_image))



X_test=np.array(data)

X_test = X_test.astype('float32')/255 

pred = model.predict_classes(X_test)
#Accuracy with the test data

from sklearn.metrics import accuracy_score

accuracy_score(y_test, pred)
# Confussion matrix 

import itertools

from sklearn.metrics import confusion_matrix



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=75) 

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

            plt.text(j, i, cm[i, j],

            horizontalalignment="center",

            color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



class_names = range(43)

cm = confusion_matrix(pred,y_test)



plt.figure(2)

plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')