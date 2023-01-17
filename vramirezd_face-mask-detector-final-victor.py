

import os

import zipfile

import random

import shutil

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from shutil import copyfile

from os import getcwd

from os import listdir

import cv2

from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from sklearn.utils import shuffle

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image  as mpimg

import shutil
#Creating the directories

os.mkdir('Temp')

os.mkdir('Temp/Mask')

os.mkdir('Temp/No Mask')



#Spliting the images

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if 'face_with_mask' in filename:

            print('image : {} '.format(filename)+' Added to the directory Mask')

            shutil.copy('/kaggle/input/mask-detection3/'+filename, '/kaggle/working/Temp/Mask')

        elif 'face_without_mask' in filename:

            print('image : {} '.format(filename)+' Added to the directory No Mask')

            shutil.copy('/kaggle/input/mask-detection3/'+filename, '/kaggle/working/Temp/No Mask')
data_path='/kaggle/working/Temp/'

categories=os.listdir(data_path)

labels = [i for i in range(len(categories))]

label_dict=dict(zip(categories,labels)) #empty dictionary



print(labels)

print(categories)

print(label_dict)

    
img_size=100

data=[]

target=[]





for category in categories:

    folder_path=os.path.join(data_path,category)

    img_names=os.listdir(folder_path)

        

    for img_name in img_names:

        img_path=os.path.join(folder_path,img_name)

        img=cv2.imread(img_path)



        try:

            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)           

            resized=cv2.resize(img,(img_size,img_size))

            #resizing the gray scale into 50x50, since we need a fixed common size for all the images in the dataset

            data.append(resized)

            target.append(label_dict[category])

            #appending the image and the label(categorized) into the list (dataset)



        except Exception as e:

            print('Exception:',e)

            #if any exception rasied, the exception will be printed here. And pass to the next image
import numpy as np



data=np.array(data)/255.0

data=np.reshape(data,(data.shape[0],img_size,img_size,3))

target=np.array(target)



from keras.utils import np_utils



new_target=np_utils.to_categorical(target)
from keras.models import Sequential

from keras.layers import Dense,Activation,Flatten,Dropout

from keras.layers import Conv2D,MaxPooling2D

from keras.callbacks import ModelCheckpoint



model=Sequential()



#The first CNN layer followed by Relu and MaxPooling layers

model.add(Conv2D(200,(3,3),input_shape=(100, 100, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



#The second convolution layer followed by Relu and MaxPooling layers

model.add(Conv2D(100,(3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



#Flatten layer to stack the output convolutions from second convolution layer

model.add(Flatten())

model.add(Dropout(0.5))



#Dense layer of 64 neurons

model.add(Dense(64,activation='relu'))



#The Final layer with two outputs for two categories

model.add(Dense(2,activation='softmax'))



model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
from sklearn.model_selection import train_test_split



train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.2,random_state=42)
plt.figure(figsize=(10,5))

pie = [len(os.listdir('/kaggle/working/Temp/Mask')),len(os.listdir('/kaggle/working/Temp/No Mask'))]

plt.pie(pie,

        labels = ['Mask','No Mask'],

        autopct='%1.1f%%'       

       )

plt.title('Proportion of each observed category')
def display_examples(class_names, images, labels):

    """

        Display 25 images from the images array with its corresponding labels

    """

    

    fig = plt.figure(figsize=(10,10))

    fig.suptitle("Some examples 25 images of the dataset", fontsize=16)

    for i in range(3):

        plt.subplot(3,1,i+1)

        plt.xticks([])

        plt.yticks([])

        plt.grid(False)

        plt.imshow(images[i], cmap=plt.cm.binary)

        plt.xlabel(class_names[labels[i]])

    plt.show()
display_examples(categories,test_data,test_target)
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

history=model.fit(train_data,train_target,epochs=20,callbacks=[checkpoint],validation_split=0.2)
# if you want to save the model to use it later uncomment the line below

#model.save('my_model.h5')



#to load it later uncomment the lines below

#from keras.models import load_model

#new_model = tf.keras.models.load_model('PATH') PATH represent the path of the saved model
def plot_accuracy_loss(history):

    """

        Plot the accuracy and the loss during the training of the nn.

    """

    fig = plt.figure(figsize=(18,8))



    # Plot accuracy

    plt.subplot(221)

    plt.plot(history.history['accuracy'],'bo--', label = "acc")

    plt.plot(history.history['val_accuracy'], 'ro--', label = "val_acc")

    plt.title("train_acc vs val_acc")

    plt.ylabel("accuracy")

    plt.xlabel("epochs")

    plt.legend()



    # Plot loss function

    plt.subplot(222)

    plt.plot(history.history['loss'],'bo--', label = "loss")

    plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")

    plt.title("train_loss vs val_loss")

    plt.ylabel("loss")

    plt.xlabel("epochs")



    plt.legend()

    plt.show()

plot_accuracy_loss(history)
predictions = model.predict(test_data)     # Vector of probabilities

pred_labels = np.argmax(predictions, axis = 1) # We take the highest probability


def display_random_image(class_names, images, labels):

    """

        Display a random image from the images array and its correspond label from the labels array.

    """

    

    index = np.random.randint(images.shape[0])

    plt.figure()

    plt.imshow(images[index])

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.title('Image #{} : '.format(index) + class_names[labels[index]])

    plt.show()





display_random_image(categories, test_data, test_target)
def print_mislabeled_images(class_names, test_images, test_labels, pred_labels):

    """

        Print 25 examples of mislabeled images by the classifier, e.g when test_labels != pred_labels

    """

    BOO = (test_labels == pred_labels)

    mislabeled_indices = np.where(BOO == 0)

    mislabeled_images = test_images[mislabeled_indices]

    mislabeled_labels = pred_labels[mislabeled_indices]



    title = "Some examples of mislabeled images by the classifier:"

    display_examples(class_names,  mislabeled_images, mislabeled_labels)

print_mislabeled_images(categories, test_data, test_target, pred_labels)