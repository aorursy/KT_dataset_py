# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D

from tensorflow.python.keras.optimizers import RMSprop

from tensorflow.python.keras.callbacks import ReduceLROnPlateau

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
# import train dataset

dftrain = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

# import test dataset

dftest = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
dftrain.shape
dftrain.head()
dftest.shape
dftest.head()
X_train = dftrain.iloc[:,1:].values

y_train = dftrain.iloc[:,0].values
X_train = X_train.reshape(X_train.shape[0], 28, 28)



for i in range(11, 14):

    plt.subplot(330 + (i+1))

    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

    plt.title(y_train[i]);
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_train.shape
#normalize x train

X_train = X_train/255
from keras.utils.np_utils import to_categorical

#to_categorical(df_label, number_of_classes)

y_train = to_categorical(y_train)

y_train.shape
y_train[6:10]
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=101)
model = Sequential()

model.add(Conv2D(60, kernel_size=(5, 5),

                 activation='relu',

                 input_shape=(28, 28, 1)))

#model.add(Dropout(0.5))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(Conv2D(40, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D())

model.add(Conv2D(40, kernel_size=(3, 3), activation='relu'))

model.add(Conv2D(40, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax'))
#Image Augmentation

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





datagen.fit(X_train)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])
# Fit the model

history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=80),

                              epochs = 30, 

                              verbose = 2, steps_per_epoch=X_train.shape[0] // 80,

                              validation_data = (X_val,y_val),

                              callbacks=[learning_rate_reduction])
predict_y = model.predict(X_val)

#df_compare = pd.DataFrame({"Actual": y_val, "Predicted": predict_y})

#df_compare.sample(10)

#predict_y

Y_pred_classes = np.argmax(predict_y,axis = 1)

Y_actual = np.argmax(y_val,axis = 1)

df_compare = pd.DataFrame({"Actual": Y_actual, "Predicted": Y_pred_classes})

df_compare.sample(10)
from sklearn.metrics import confusion_matrix

import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(predict_y,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
X_test = dftest.values

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_test = X_test/255

X_test.shape
predictions = model.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})

submissions.to_csv("DRv11.csv", index=False, header=True)
submissions.head(10)