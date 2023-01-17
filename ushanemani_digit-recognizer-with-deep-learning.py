# Import all Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as img
%matplotlib inline

#import Keras Libs

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

sns.set(style='white', context ='notebook', palette = 'deep')

import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
y_train = train['label']
X_train = train.drop('label',axis=1)
del train
sns.countplot(y_train)
y_train.value_counts()
#check for nulls in Xtrain
X_train.isnull().any().describe()
test.isnull().any().describe()
# Normalise the data
X_train = X_train/255.0
test =test/255.0
#Train and test images (28px x 28px) has been stock into pandas.Dataframe as 1D vectors of 784 values. We reshape all data to 28x28x1 3D matrices.
#Keras requires an extra dimension in the end which correspond to channels. 
#MNIST images are gray scaled so it use only one channel.
#For RGB images, there is 3 channels, we would have reshaped 784px vectors to 28x28x3 3D matrices.
#reshape the data to 28 by 28 by 1  - 3-D

X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

y_train = to_categorical(y_train, num_classes=10) # one hot encoding to 0,0,1,0,0,0 
# split train data into train n test
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
X_train2, X_cv, y_train2, y_cv = train_test_split(X_train, y_train, test_size = 0.20, random_state=42)
# create Conv Neural Net model
# Initialising the CNN
classifier = Sequential()
#Layer -1 Conv, Conv, Maxpooling 
classifier.add(Conv2D(32, (5, 5), input_shape = (28, 28, 1), activation = 'relu',padding ='Same'))
classifier.add(Conv2D(32, (5, 5), activation = 'relu',padding ='Same'))
classifier.add(MaxPool2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))

#Layer -2 Conv, Conv, Maxpooling 
classifier.add(Conv2D(64, (3, 3), activation = 'relu',padding ='Same'))
classifier.add(Conv2D(64, (3, 3), activation = 'relu',padding ='Same'))
classifier.add(MaxPool2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))

# Full connection - Dense used here
classifier.add(Flatten())
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 10, activation = 'softmax'))

#define optmizer
optmizer = RMSprop(lr=0.001,rho=0.9, epsilon = 1e-08, decay =0.0)
classifier.compile(optimizer=optmizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
from keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor ='val_acc', patience=3, verbose=1, factor =0.5, min_lr =0.00001)
epochs=5
batch_size=86
# without Data Augmentation
#history = classifier.fit(X_train2,y_train2, batch_size=batch_size,epochs =epochs, validation_data=(X_cv,y_cv), verbose=2)
#with Data Augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train2)
history = classifier.fit_generator(datagen.flow(X_train2,y_train2, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_cv,y_cv),
                              verbose = 2, steps_per_epoch=X_train2.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])
#predict results
results = classifier.predict(test)
results =  np.argmax(results, axis=1)  # select indices with max probability
results = pd.Series(results, name ='Label')
#write to file
submission = pd.concat([pd.Series(range(1,28001),name='ImageId'),results],axis=1)
submission.to_csv('cnn_mnist2.csv', index=False)
