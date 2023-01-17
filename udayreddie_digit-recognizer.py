# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

config = tf.ConfigProto()

config.gpu_options.allow_growth=True

sess = tf.Session(config=config)
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.head()
test.head()
X_train = np.array(train.iloc[:,1:])

y_train = np.array(train.iloc[:,0])



X_test = np.array(test.iloc[:,1:])

y_test = np.array(test.iloc[:,0])



print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
import matplotlib.pyplot as plt    # For plotting 

%matplotlib inline                 



s = np.random.choice(range(X_train.shape[0]), size=10, replace=False)  # Randomly select few samples



print(s)

plt.figure(figsize=(15,5))

for i,j in enumerate(s):   

    plt.subplot(2,5,i+1)                                # Subplot flag

    plt.imshow(np.array(X_train[j]).reshape(28,28))     # Plot the image

    plt.title('Product: '+str(y_train[j]))              # Target of the image

    plt.xticks([])                                      # No X-Axis ticks

    plt.yticks([])                                      # No Y-Axis ticks

    plt.gray()   
# Check the dimensions of the arrays

print('x_train shape: {}'.format(X_train.shape))

print('y_train shape: {}'.format(y_train.shape))

print('x_test shape:  {}'.format(X_test.shape))

print('y_test shape:  {}'.format(y_test.shape))
# 'to_categorical' converts the class lebels to one-hot vectors. One-hot vector is nothing but dummifying in R.

from keras.utils import to_categorical

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
# Sequential is a container which stores the layers in order. 

# Think of it as a train engine to which you can keep adding train cars. train car in our context will be a layer.

# 'Dense' is a fully connected layer feedforward layer.

from keras.models import Sequential 

from keras.layers import Dense

from keras import regularizers, optimizers
# Building a simple MLP



model = Sequential() # This initializes a sequential model to which we can keep adding layers.

model.add(Dense(10,input_shape=(784,), kernel_initializer='uniform', 

                activation='softmax')) # Add output layer
# Setting learning and momentum

# Adam is the optimizer which is the state of the art Gradient Descent variation. 

from keras.optimizers import Adam

adam = Adam(lr=0.001)



model.compile(loss='categorical_crossentropy', # CrossEntropy is the loss function. 

              optimizer=adam,                  # Mention the optimizer

              metrics=['accuracy'])            # Mention the metric to be printed while training
nb_epochs = 50

# training the MLP model

history = model.fit(X_train, y_train, epochs=nb_epochs, batch_size=64, 

                    validation_split=0.1) 
train_acc = history.history['acc']

train_loss = history.history['loss']



val_acc = history.history['val_acc']

val_loss = history.history['val_loss']
from matplotlib import pyplot as plt #plt is a visualization module in matplotlib.  

%matplotlib inline 

plt.figure(figsize=(20,5))

plt.subplot(1,2,1)

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.plot(train_loss)

plt.plot(val_loss)



plt.subplot(1,2,2)

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.plot(train_acc)

plt.plot(val_acc)
# Building a simple MLP



model = Sequential() # This initializes a sequential model to which we can keep adding layers.

model.add(Dense(200, kernel_initializer='uniform', 

                input_dim = 784, activation='tanh')) # Add a dense layer 

model.add(Dense(10, kernel_initializer='uniform', 

                activation='softmax')) # Add output layer





# Setting learning and momentum

# Adam is the optimizer which is the state of the art Gradient Descent variation. 

from keras.optimizers import Adam

adam = Adam(lr=0.001)



model.compile(loss='categorical_crossentropy', # CrossEntropy is the loss function. 

              optimizer=adam,                  # Mention the optimizer

              metrics=['accuracy'])            # Mention the metric to be printed while training





nb_epochs = 50

# training the MLP model

history = model.fit(X_train, y_train, epochs=nb_epochs, batch_size=64, 

                    validation_split=0.1) 



train_acc = history.history['acc']

train_loss = history.history['loss']



val_acc = history.history['val_acc']

val_loss = history.history['val_loss']





from matplotlib import pyplot as plt #plt is a visualization module in matplotlib.  

%matplotlib inline 

plt.figure(figsize=(20,5))

plt.subplot(1,2,1)

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.plot(train_loss)

plt.plot(val_loss)



plt.subplot(1,2,2)

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.plot(train_acc)

plt.plot(val_acc)
model.summary()
# predict results

results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("Digit_Rec_Sub.csv",index=False)