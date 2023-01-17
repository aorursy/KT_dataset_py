# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.getcwd())

print(os.listdir("../input"))

os.chdir('../input/cell_images/cell_images')

# Any results you write to the current directory are saved as output.
import cv2
DATADIR = os.getcwd()



print(os.listdir(DATADIR))



CATEGORIES  = ['Parasitized','Uninfected']

print(DATADIR)
def getdata():

    data = []

    for category in CATEGORIES:

        path = os.path.join(DATADIR,category)

        for image in os.listdir(path):

            try:

                img_array = cv2.imread(os.path.join(path,image))

                new_array = cv2.resize(img_array,(50,50))

                data.append([new_array,CATEGORIES.index(category)])

            except Exception as e:

                pass

            

    return data
data = []

data[:] = getdata()

import random

random.shuffle(data)

data = pd.DataFrame(data)

print(data.head())
data[1]
X=[]

Y=[]

X[:] = data[0]

Y[:] = data[1]



X=np.array(X)

X.shape
X = X/255
X.shape[1:]
import tensorflow as tf

from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Activation,Flatten

from tensorflow.keras.models import Sequential
model = Sequential()



model.add(Conv2D(64,(3,3),input_shape = X.shape[1:] ))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Conv2D(64,(3,3)) )

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Conv2D(64,(3,3)) )

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Flatten())

model.add(Dense(64))



model.add(Dense(1))

model.add(Activation("sigmoid"))
model.compile(loss = "binary_crossentropy",

             optimizer = "adam",

             metrics = ['accuracy'])

model.fit(X,Y,batch_size = 32,validation_split=0.2,epochs=3)




os.chdir("../../../working")

print(os.getcwd())

os.listdir("/kaggle")
model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model.h5")

print("Saved model to disk")
