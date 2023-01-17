# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = '/kaggle/input/breast-cancer-wisconsin-data/data.csv' #Set the path variable which contains the input dataset
dataset = pd.read_csv(path)
print(dataset.shape)
print(dataset.head())
print(dataset.tail())
from sklearn.preprocessing import normalize
dataset = dataset.iloc[:,1:32]
X=dataset.iloc[:,1:]
X=(X-X.min())/(X.max()-X.min())*10
Y=dataset.iloc[:,0]
label_to_number=lambda x:1 if x=='M'else 0
Y=Y.apply(label_to_number)
print(X)
print(Y)
m=X.shape[0] #Number of training examples
train_m,test_m = int(0.8*m),int(0.2*m) #split the dataset into 2 sets
X_train = X[:train_m]
Y_train = Y[:train_m]
X_test = X[train_m:]
Y_test = Y[train_m:]
print(X_train.shape)
print(X_test.shape)
print(m)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model=keras.Sequential([
    layers.Dense(10,input_shape=(X_train.shape[1],),activation='relu'),
    layers.Dense(15,activation='relu'),
    layers.Dense(1,activation= 'sigmoid')
])
model.summary()
class myCallbacks(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        acc = logs.get("accuracy")
        if acc>0.99:
            self.model.stop_training=True
        
callbacks=myCallbacks()
    
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=1000,callbacks=[callbacks])
model.evaluate(X_test,Y_test)