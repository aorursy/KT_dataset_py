#importing necessary modules

import numpy as np

import glob

import matplotlib.pyplot as plt

import path

import os

import cv2
#loading images directory based on image format example below lines take extension '.PNG'

Parasitized_images= glob.glob('/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/*.png')

Uninfected_images= glob.glob('/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/*.png')
img_list=[] #creating empty lst
# converting images to numerical and resizing and appending the data to empty list 

for img in Parasitized_images:

    img=cv2.imread(img)#reading image and storing numerical data to img variable

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)# converting color 

    img=cv2.resize(img,(50,50))#resizing image to (50,50)

    img_list.append(img)

    

    

for img in Uninfected_images:

    img=cv2.imread(img)

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img=cv2.resize(img,(50,50))

    img_list.append(img) 
img_array=np.array(img_list)# converting list to array
img_array=img_array.astype('float32')#converting numerical data to floating point
img_array = img_array/ 255 # normalizing or scaling images value between 0 to 1
print(img_array.dtype)

print(img_array.shape)
y=np.ones((img_array.shape[0],),dtype='int64')#creating an array of 27558 with all values of one

y.shape
y[0:13778]=0 #making y values to 0 from 0 to 13778

y[13778:]=1
x=img_array #storing the data to another variable
#importimg module

from sklearn.utils import shuffle 
#shuffling the data

x,y=shuffle(x,y,random_state=42)
class_labels_dictionary={0:'Parasitized',1:'Uninfected'}
# displaying sample images

for i in range(0,10):

    image=x[i]

    img_label_value=y[i]

    img_class_name=class_labels_dictionary[img_label_value]

    # Draw the image as a plot

    plt.imshow(image)

    # Label the image

    plt.title(img_class_name)

    # Show the plot on the screen



    plt.axis('off')

    plt.show()



'''i=int(input('enter any number between 0 to 27558 : '))

while True:

    image=x[i]

    img_label_value=y[i]

    img_class_name=class_labels_dictionary[img_label_value]

    # Draw the image as a plot

    plt.imshow(image)

    # Label the image

    plt.title(img_class_name)

    # Show the plot on the screen



    plt.axis('off')

    plt.show()

    break

'''



y
from keras.utils import np_utils

number_of_classes=2

# label encoding

y=np_utils.to_categorical(y,number_of_classes)
y
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)# splitting data for train and test
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from pathlib import Path

from keras.layers import Activation

from keras.layers import BatchNormalization
# creating a network

model = Sequential()



model.add(Conv2D(32, (3, 3), padding='same', input_shape=(50,50,3), activation="relu"))

model.add(Conv2D(64, (3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))

model.add(Conv2D(64, (3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.3))





model.add(Flatten())

model.add(Dense(512, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(2, activation="sigmoid"))

model.summary()
#loss and optimising functions

model.compile(

    loss='binary_crossentropy',

    optimizer='adam',

    metrics=['accuracy']

)

# Train the model

hist=model.fit(

    x_train,

    y_train,

    batch_size=200,

    epochs=17,

    validation_data=(x_test, y_test),

    shuffle=True

)

# train and test accuracy graph

plt.figure(figsize = (20,10))

plt.plot(hist.history['accuracy'], label = 'train_accuracy')

plt.plot( hist.history['loss'], label = 'train_loss')

plt.xlabel("Number of Epoch's")

plt.ylabel('Accuracy/Loss ')

plt.title('train_accuracy and train_loss')

#evaluating loss and accuracy for test dataset

predictions=model.evaluate(x_test,y_test)


print(f'accuracy : {predictions[1]}')

print(f'loss : {predictions[0]}')
model.save('CNN_malaria_weights.model') #saving model to file
new_model=keras.models.load_model('CNN_malaria_weights.model') #loading model from created file
pred=new_model.predict_classes(x_test) # predicting the data
from sklearn.metrics import classification_report,confusion_matrix

import scipy
print(confusion_matrix(np.argmax(y_test,axis=1),pred))
print(classification_report(np.argmax(y_test,axis=1),pred))
classes_dic={0:'Parasitized',1:'Uninfected'}
img='../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/C100P61ThinF_IMG_20150918_144104_cell_128.png'

def fun(classes_dic,img):

    img=scipy.misc.imread(img)# reading image data and storing in variable

    imge=scipy.misc.imresize(img,(50,50))# resizing to 50,50 

    img=imge.astype('float32')#converting numerical to floating point

    img /= 255# scaling data 0 to 1

    img=img.reshape(1,50,50,3)# reshaping image 

    clas=new_model.predict(img)# predicting

    clas=np.argmax(clas)# takes max value in each row  

    plt.title(classes_dic[clas])

    plt.imshow(imge)

fun(classes_dic,img)   
