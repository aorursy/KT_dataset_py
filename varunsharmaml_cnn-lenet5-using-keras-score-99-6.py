import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model

from keras.utils import to_categorical
from keras.initializers import glorot_uniform

import matplotlib.pyplot as plt
%matplotlib inline

import pandas as pd
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
data = pd.read_csv('../input/train.csv')
data.shape
train = data[:39000]
dev = data[39000:]
train.shape, dev.shape
train = train.as_matrix()
dev = dev.as_matrix()
train.shape, dev.shape
def give_data(set):
    Y = set[:,0]
    Y = to_categorical(Y, num_classes=10)
    X = set[:,1:].reshape((set.shape[0],28,28,1))/255.
    return X,Y
X_dev, Y_dev = give_data(dev)
X_dev.shape, Y_dev.shape
X_train, Y_train = give_data(train)
X_train.shape, Y_train.shape
def LeNet5(input_shape = (28,28,1), classes = 10):
    X_input = Input(input_shape)
    
    X = ZeroPadding2D((1,1))(X_input)
    X = Conv2D(8, (3,3), name='Conv1')(X)
    X = BatchNormalization(axis=3, name='BatchNorm1')(X)
    
    X = MaxPooling2D(strides=(2,2), name='MaxPool1')(X)
    
    X = Conv2D(16, (5,5), name='Conv2')(X)
    
    X = Flatten()(X)
    X = Dense(120, activation='relu', name='fc1')(X)
    X = Dense(84 , activation='relu', name='fc2')(X)
    
    X = Dense(classes, activation='softmax', name='final_layer')(X)
    
    model = Model(inputs=X_input, outputs=X, name='LeNet5')
    
    return model
model = LeNet5()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
track = model.fit(X_train, Y_train, batch_size=64, epochs=30, validation_data=(X_dev, Y_dev))
loss = track.history["loss"]
acc = track.history["acc"]
ep = list(range(len(loss)))
plt.plot(ep, loss)
plt.xlabel("#epochs")
plt.ylabel("loss")
plt.plot(ep, acc)
plt.xlabel("#epochs")
plt.ylabel("accuracy")
model.summary()
test = pd.read_csv("../input/test.csv")
test = test.as_matrix()
test.shape
X_test = test.reshape((28000,28,28,1))
X_test.shape
pred = model.predict(X_test)
pred.shape
res = pred.argmax(axis=1)
res.shape
index = 5
print("Prediction :",res[index])
plt.imshow(X_test[index].reshape((28,28)), cmap="gray")
