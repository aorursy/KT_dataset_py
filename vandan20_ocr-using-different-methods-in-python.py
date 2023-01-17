import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from keras.layers import Conv2D,Dense,MaxPool2D,Dropout,Flatten

from keras.models import Sequential

from skimage import io

import cv2

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from matplotlib.image import imread

img=imread('/kaggle/input/street-view-getting-started-with-julia/train/train/1.Bmp')

plt.imshow(img)
train=[io.imread('/kaggle/input/street-view-getting-started-with-julia/train/train/'+str(i)+'.Bmp',as_gray=True) for i in range(1,6283)]

plt.imshow(train[1])
final_train=[cv2.resize(image,(28,28)).flatten() for image in train]
labels=pd.read_csv('/kaggle/input/street-view-getting-started-with-julia/trainLabels.csv')

labels.head()
finaly=np.array(final_train)

finaly.shape
mapp={}

a='abcdefghijklmnopqrstuvwxyz'

count=0

for x in range(10):

    mapp[x]=count

    count+=1

for y in a:

    mapp[count]=y.upper()

    count+=1

for y in a:

    mapp[count]=y

    count+=1

trainx,testx,trainy,testy=train_test_split(finaly,labels['Class'].iloc[:-1],test_size=0.10)
model=KNeighborsClassifier(n_neighbors=1,n_jobs=-1)

model.fit(trainx,trainy)
model.score(testx,testy)
features_for_conv=np.array([cv2.resize(image,(28,28)) for image in train])

features_for_conv=features_for_conv[:,:,:,np.newaxis]
cmodel=Sequential()

cmodel.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1),data_format='channels_last'))

cmodel.add(MaxPool2D((2,2)))

cmodel.add(Dropout(0.25))

cmodel.add(Conv2D(64,(3,3),activation='relu'))

cmodel.add(MaxPool2D((2,2)))

cmodel.add(Dropout(0.25))

cmodel.add(Flatten())

cmodel.add(Dense(128,activation='relu'))

cmodel.add(Dense(62,activation='softmax'))
Labels=pd.get_dummies(labels['Class'])
Trainx,valx,Trainy,valy=train_test_split(features_for_conv,Labels.iloc[:-1],test_size=0.2)
Trainx.shape
cmodel.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])
cmodel.fit(Trainx,Trainy,epochs=15,validation_data=(valx,valy))
cmodel.fit(features_for_conv,Labels.iloc[:-1],epochs=14)
lab=[]

test=[]

import os

for i in os.listdir('/kaggle/input/street-view-getting-started-with-julia/test/test/'):

    test.append(io.imread('/kaggle/input/street-view-getting-started-with-julia/test/test/'+i,as_gray=True))

    lab.append(i.split('.')[0])

test_img=np.array([cv2.resize(image,(28,28)) for image in test])

test_img=test_img[:,:,:,np.newaxis]

test_img.shape
predictions=cmodel.predict(test_img)
predictions=np.argmax(predictions,axis=1)
lit=[]

for x in predictions:

    lit.append(mapp.get(x))

lit
sub=pd.DataFrame({'ID':lab,'Class':lit})
sub.to_csv('julia.csv')