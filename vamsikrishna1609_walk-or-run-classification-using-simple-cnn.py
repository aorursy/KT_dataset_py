import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

import cv2

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix





from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

import itertools





sns.set(style='white', context='notebook', palette='deep')

n = 128

i=0

X=np.zeros((741,n,n))

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        im= cv2.imread((os.path.join(dirname, filename)))

        resized_image = cv2.resize(im, (n,n))

        X[i,:,:]=np.sum(resized_image,axis=2)

        i+=1
X/=255*3 # normalising 

X = X.reshape(-1,n,n,1) #reshaping as tensor becuase keras accpets only tensor inputs

X.shape
train_label = np.zeros(741)

label=np.array([82,59,299,301])

a=0

for i in range(4):

    for j in range(label[i]):

        train_label[j+a]=i

    a+=j+1

labels=train_label%2
labels
X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size = 0.2, random_state=50)
model = Sequential()



model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu', input_shape = (128,128,1)))

model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.01))





model.add(Conv2D(filters = 8, kernel_size = (1,1),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 8, kernel_size = (1,1),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.01))





model.add(Flatten())

model.add(Dense(16, activation = "relu"))

model.add(Dropout(0.01))

model.add(Dense(1, activation = "sigmoid"))
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, decay=0.0)

model.compile(optimizer = optimizer , loss = "binary_crossentropy", metrics=["accuracy"])
epochs = 50

batch_size = 50
history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 

           verbose = 1)
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

#ax[0].plot(history.history['test_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='r', label="Training accuracy")

#ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)

# Predict the values from the validation dataset

Y_pred = model.predict(X_test)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred) 

# Convert validation observations to one hot vectors

yl=len(Y_pred)

Y_test=Y_test.reshape(yl,1)

for j in range(yl):

    if Y_pred[j]>0.25:

        Y_pred[j]=1

    else:

        Y_pred[j]=0        

confusion_matrix(Y_test, Y_pred)

score=model.evaluate(X_test,Y_test)

print(f'loss value is {score[0]})')

print(f'accuracy of the model is {score[1]*100}%')