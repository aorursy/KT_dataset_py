

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from time import time

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import VGG16

from keras.callbacks import TensorBoard,EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,Callback

# import the necessary packages

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam,SGD,RMSprop

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dropout

from keras.layers.core import Dense

from keras import backend as K

from keras.utils import to_categorical

# Any results you write to the current directory are saved as output.
datapath='../input/'

irisdf = pd.read_csv(datapath+'Iris.csv')

irisdf=irisdf.drop('Id',axis=1) 

X = irisdf.drop("Species",axis=1).values

# X = preprocessing.normalize(X, norm='l2',axis=1)#把这行反注释了，跑



# print(X.head(3))

# print(X_normalized[:3])



target_names = irisdf['Species'].unique()

target_dict = {n:i for i,n in enumerate(target_names)}

y= irisdf['Species'].map(target_dict)

print(y.head(4))

y_cat = to_categorical(y)

X_train,X_test,y_train,y_test = train_test_split(X,y_cat,test_size=0.2)

# X_normalized_train,X_normalized_test,y_normalized_train,y_normalized_test = train_test_split(X_normalized.values,y_cat,test_size=0.2)



# print(X_train)

print(y_cat[:10])
fig = irisdf[irisdf.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')

irisdf[irisdf.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)

irisdf[irisdf.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Sepal Length")

fig.set_ylabel("Sepal Width")

fig.set_title("Sepal Length VS Width")

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()
class LossHistory(Callback):

    def on_train_begin(self, logs={}):

        self.losses = {'batch': [], 'epoch': []}

        self.accuracy = {'batch': [], 'epoch': []}

        self.val_loss = {'batch': [], 'epoch': []}

        self.val_acc = {'batch': [], 'epoch': []}



    def on_batch_end(self, batch, logs={}):

        self.losses['batch'].append(logs.get('loss'))

        self.accuracy['batch'].append(logs.get('acc'))

        self.val_loss['batch'].append(logs.get('val_loss'))

        self.val_acc['batch'].append(logs.get('val_acc'))



    def on_epoch_end(self, batch, logs={}):

        self.losses['epoch'].append(logs.get('loss'))

        self.accuracy['epoch'].append(logs.get('acc'))

        self.val_loss['epoch'].append(logs.get('val_loss'))

        self.val_acc['epoch'].append(logs.get('val_acc'))



    def plot(self, loss_type):

        iters = range(len(self.losses[loss_type]))

    

        plt.figure(figsize=(16,10))

        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')

        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')

        plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')

        plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')

        plt.grid(True)

        plt.xlabel(loss_type)

        plt.ylabel('acc-loss')

        plt.legend(loc="upper right")

        plt.show()

        

    def save(self,name):

        arr=np.vstack((self.accuracy["epoch"],self.losses["epoch"],self.val_acc["epoch"],self.val_loss["epoch"]))

        np.save(name,arr)

        
from keras.initializers import normal,Zeros,Constant

# initializer=Zeros()

initializer=normal(mean=0., stddev=0.1, seed=13)#改这儿和后三行，比较不同weights

model = Sequential()

model.add(Dense(32, input_shape=(4,), activation='relu',kernel_initializer=initializer ))

model.add(Dense(3, activation='softmax',kernel_initializer=initializer))



model.compile(Adam(lr=0.1),

              loss='categorical_crossentropy',

              metrics=['accuracy'])

# For a multi-class classification problem

history = LossHistory()

                              

                              

callbacks_list = [ history]



#----

model.fit(X_train, y_train, epochs=20, validation_split=0.1,callbacks=callbacks_list)

history.plot("epoch")

history.save("A.npy")
y_pred = model.predict(X_test)

y_test_class = np.argmax(y_test, axis=1)

y_pred_class = np.argmax(y_pred, axis=1)

from sklearn.metrics import classification_report

print(classification_report(y_test_class, y_pred_class))