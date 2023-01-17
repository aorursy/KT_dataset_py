"""
importing the libraries
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

# Input data files are available in the "../input/" directory.

import os
print(os.listdir("../input"))
#getting the dataset in proper order
dataset=pd.read_csv("../input/labels.csv")
dataset_test=pd.read_csv("../input/sample_submission.csv")
#print(len(labels),labels.head()) -----> 10222 and two columns id and breed
#print(len(dataset_test)) ------> 10357 rows
#os.listdir("../input/train/")
target=pd.Series(dataset['breed'])

#One hot Encoding
one_hot=pd.get_dummies(target,sparse=True)
target=np.asarray(one_hot)
#print(target[12])-----> 
"""it store a particular value into a series of array which has value for all
irrelevant classes and 1 for the relevant class
"""


#Now Converting the images to numpy arrays both for test and for train
im_size=100 #----> image size
ids=dataset['id']
X_train=[]
y_train=target
#print(y_train[1].shape)
for id in ids:
    img=cv2.imread('../input/train/{}'.format(id)+'.jpg')
    X_train.append(cv2.resize(img,(im_size,im_size)))
X_train=np.array(X_train,np.float32)
#Dataset _---> Continue #Test Data
#print(X_train.shape)
X_test=list()
ids_test=dataset_test['id']
for id in ids_test:
    img=cv2.imread('../input/test/{}'.format(id)+'.jpg')
    X_test.append(cv2.resize(img,(im_size,im_size)))
X_test=np.array(X_test,np.float32) #----> Competition test data
#Dataset ----> Continue (Standardize the image data)
#print(X_test[1])
def standardize(array):
    array/=255
    return array
X_train=standardize(X_train)
X_test=standardize(X_test)
#print(X_train[1])
#print(X_test[1])
"""
Declaring Model variables like batch_size, number of iterations or epochs and declaring first
architecture for the following image classification problem
"""
#print(dataset_test.columns) #----> 121 columns and first one is id and others are species or number
#classes so number of classes are 120
num_classes=120
batch_size=32 #Randomly
epochs=50 #Randomly

#Now importing important libraries and respective methods
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
print(X_train.shape,y_train.shape,X_test.shape)
#defining the model architecture
"""
For the first model I will take architecture conv layers followed by pooling layers
model 1st
"""
def mode_architecture():
    model=Sequential()
    model.add(Conv2D(32,(3,3),input_shape=X_train.shape[1:],activation='relu',padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2))
    #model.add(Dropout(0.20))
    
    model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2))
    #model.add(Dropout(0.25))
    
    model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2))
    
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
arch_cnn=mode_architecture()
arch_cnn.summary()
#Creation of validation set from training observations
#from sklearn.model_selection import train_test_split
#train_x,test_x,train_y,test_y=train_test_split(X_train,y_train,test_size=0.3,random_state=0)

#Fitting the model
val_arch=arch_cnn.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1)
test_predictions=arch_cnn.predict(X_test,batch_size=32,verbose=1)
#Now creating the submission file
dog_species=dataset_test.columns[1:]
submission_shani=pd.DataFrame(data=test_predictions,index=ids_test,columns=dog_species)
submission_shani.index.name='id'
submission_shani.to_csv('submission_shani.csv',index=True,encoding='utf-8')

