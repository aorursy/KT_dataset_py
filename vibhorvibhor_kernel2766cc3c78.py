import matplotlib.pyplot as plt

import pandas as pd

from keras.utils import np_utils

from keras.initializers import he_normal

import seaborn as sns

from keras.models import Sequential

from keras.layers import Dense , Activation

from keras import optimizers

from keras import losses

import numpy as np

import pandas as pd

import time

X_train = pd.read_csv(r'/kaggle/input/64-bit-puf-xor/XOR_Arbiter_PUFs/6xor_64bit/train_6xor_64dim.csv',header = None)





#distrb = X_train.iloc[:,64].value_counts()



Y_train = X_train[[64]]

X_train.drop([64],axis = 1,inplace = True)



X_test = pd.read_csv(r'/kaggle/input/64-bit-puf-xor/XOR_Arbiter_PUFs/6xor_64bit/test_6xor_64dim.csv',header = None)

X_test.head()

Y_test = X_test[[64]]

X_test.drop([64],axis = 1,inplace = True)

y_train = np_utils.to_categorical(Y_train, 2) 

y_test = np_utils.to_categorical(Y_test, 2)

print(X_train.describe())



print(X_train.shape)



nepoch = 30

outlayer = 2

batch_size = 10000

from keras.layers.normalization import BatchNormalization

from keras.layers import Dropout

from keras.layers.merge import concatenate

from keras.utils import plot_model

from keras.layers import Input

from keras.models import Model



input_layer = Input(shape = (64,))

model = Sequential()

model.add(Dense(100,input_dim=64,activation = 'sigmoid'))

model.add(Dense(64,input_dim=64,activation = 'sigmoid'))

model.add(Dense(20,input_dim=64,activation = 'sigmoid'))



model.add(Dense(2,activation = 'softmax'))









# summarize layers

print(model.summary())



# plot graph

plot_model(model, to_file='MODEL.png')



adam = optimizers.Adam(lr = 0.001)

model.compile(loss=losses.categorical_crossentropy, optimizer = adam, metrics=['accuracy'])

start = time.clock()

hist = model.fit(X_train, y_train, epochs=nepoch, batch_size=batch_size,validation_data = (X_test,y_test))

# Final evaluation of the model

scores = model.evaluate(X_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))

print(time.clock() - start)

'''



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

        

        

# Any results you write to the current directory are saved as output.

'''
