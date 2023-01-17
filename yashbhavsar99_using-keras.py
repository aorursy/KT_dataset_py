# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import tensorflow.keras as keras

from keras.utils import to_categorical



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import csv

X_train = []

Y_train =[]

#rows=[]

csv_file =  open('/kaggle/input/digit-recognizer/train.csv') 

csvreader = csv.reader(csv_file)

#fields = csvreader.next()

i=0

for row in csvreader:

    i=i+1

    if i==1:

        continue

    row = np.array(row)

    label =int(row[0])

    row = np.reshape(row[1:],[28,28])

    X_train.append(row)

    Y_train.append(label)

    



X_train = np.array(X_train)

X_train = np.reshape(X_train,[X_train.shape[0],X_train.shape[1],X_train.shape[2],1])

Y_train = np.array(Y_train)

Y_train = np.reshape(Y_train, [Y_train.shape[0],1])

print(X_train.shape,Y_train.shape)



Y_train = np.reshape(Y_train,[Y_train.shape[0],1])

#Y_train = int(Y_train)

#Y_train = to_categorical(Y_train,num_classes=10)

print(Y_train.shape)

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
model = Sequential()

model.add(Conv2D(32,(3,3),activation = 'relu',input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=(2,2),strides=2))

model.add(Flatten())

model.add(Dense(100,activation='relu',))

model.add(Dense(10,activation='softmax'))

opt= keras.optimizers.SGD(lr=0.00005,momentum=0.95)

model.compile(optimizer = opt,loss = 'categorical_crossentropy',metrics=['accuracy'])

def tocategorical(Y,num_classes):

    m = Y.shape[0]

    temp = np.zeros([m,num_classes],dtype=float)

    for i in range(m):

        temp[i,Y[i]]=1

    return temp
print(Y_train[6])

Y_train = tocategorical(Y_train,num_classes=10)

print(Y_train[6])
print(Y_train.shape) 
print(X_train.shape,Y_train.shape)

print(type(X_train[1][1][1][0]))

X_train = X_train.astype(np.float)

X_train_new = X_train[:41000,:,:,:]

Y_train_new = Y_train[:41000,:]

X_val = X_train[41000:,:,:,:]

Y_val = Y_train[41000:,:]

model.fit(X_train,Y_train,validation_data=(X_val,Y_val), epochs=10)

model.save_weights('weights_with_10_epochs.h5')
model.predict(X_train[32:35,:,:,:])
import csv

X_test = []

#rows=[]

csv_file =  open('/kaggle/input/digit-recognizer/test.csv') 

csvreader = csv.reader(csv_file)

#fields = csvreader.next()

i=0

for row in csvreader:

    i=i+1

    if i==1:

        continue

    row = np.array(row)

    #label =int(row[0])

    row = np.reshape(row,[28,28])

    X_test.append(row)

    #Y_test.append(label)



#print(X_test[0])
X_test = np.array(X_test)

X_test = X_test.astype(np.float)

X_test = np.reshape(X_test,[X_test.shape[0],X_test.shape[1],X_test.shape[2],1])
X_test.shape
Y_pred = model.predict(X_test)

print(Y_pred.shape)
print(Y_test.shape)
Y_test = np.zeros([X_test.shape[0]],int)

for i in range(Y_pred.shape[0]):

    Y_test[i] = np.argmax(Y_pred[i,:])

   

#print(Y_test[1:1200])
#print(Y_test[1:120],Y_test[120:250])
with open('my_submission.csv', 'w', newline='') as file:

    writer = csv.writer(file)

    writer.writerow(["ImageID", "Label"])

    for i in range(Y_test.shape[0]):

        writer.writerow([i+1,Y_test[i]])

    
!ls