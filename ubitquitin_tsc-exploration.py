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
train = pd.read_csv('/kaggle/input/heartbeat/mitbih_train.csv',header=None)

test = pd.read_csv('/kaggle/input/heartbeat/mitbih_test.csv',header=None)
print(train.shape)

print(test.shape)
#plot first for training heartbeats. Each heartbeat is 188 long.

import matplotlib.pyplot as plt



plt.subplot(2,2,1)

plt.plot(train.iloc[0,:186])



plt.subplot(2,2,2)

plt.plot(train.iloc[1,:186])



plt.subplot(2,2,3)

plt.plot(train.iloc[2,:186])



plt.subplot(2,2,4)

plt.plot(train.iloc[3,:186])



print(train[187][0], train[187][1], train[187][2], train[187][3])
print(train[187].value_counts())
f, axs = plt.subplots(5,1,figsize=(5,10))



plt.subplot(5,1,1)

plt.ylabel("Normal")

plt.ylim(0,1)

plt.plot(train.loc[train[187] == 0.0].loc[0])



plt.subplot(5,1,2)

plt.ylabel("Supraventricular Premature")

plt.ylim(0,1)

plt.plot(train.loc[train[187] == 1.0].loc[72471])



plt.subplot(5,1,3)

plt.ylabel("Premature VC")

plt.ylim(0,1)

plt.plot(train.loc[train[187] == 2.0].loc[74694])



plt.subplot(5,1,4)

plt.ylabel("Fusion")

plt.ylim(0,1)

plt.plot(train.loc[train[187] == 3.0].loc[80482])



plt.subplot(5,1,5)

plt.ylabel("Unclassifiable Beat")

plt.ylim(0,1)

plt.plot(train.loc[train[187] == 4.0].loc[81123])
train_target = train[187]

label= 187



df = train.groupby(label, group_keys=False)

train = pd.DataFrame(df.apply(lambda x: x.sample(df.size().min()))).reset_index(drop=True)
print(train[187].value_counts())
train_target = train[187]

train_target = train_target.values.reshape(3205,1)

#one hot encode train_target



from sklearn.preprocessing import OneHotEncoder

from sklearn import preprocessing

# TODO: create a OneHotEncoder object, and fit it to all of X



# 1. INSTANTIATE

enc = preprocessing.OneHotEncoder()



# 2. FIT

enc.fit(train_target)



# 3. Transform

onehotlabels = enc.transform(train_target).toarray()

onehotlabels.shape



target = onehotlabels
print(target[0])
from sklearn.model_selection import train_test_split



X = train

X = X.drop(axis=1,columns=187)



X_train, X_valid, Y_train, Y_valid = train_test_split(X,target, test_size = 0.25, random_state = 36)
X_train = np.asarray(X_train)

X_valid = np.asarray(X_valid)

Y_train = np.asarray(Y_train)

Y_valid = np.asarray(Y_valid)



#X_train.reshape((1, 2403, 187))

X_train = np.expand_dims(X_train, axis=2)

X_valid = np.expand_dims(X_valid, axis=2)

print(X_train.shape)

print(Y_train.shape)
#1. Function to plot model's validation loss and validation accuracy

def plot_model_history(model_history):

    fig, axs = plt.subplots(1,2,figsize=(15,5))

    # summarize history for accuracy

    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])

    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy')

    axs[0].set_xlabel('Epoch')

    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)

    axs[0].legend(['train', 'val'], loc='best')

    # summarize history for loss

    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])

    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])

    axs[1].set_title('Model Loss')

    axs[1].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')

    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)

    axs[1].legend(['train', 'val'], loc='best')

    plt.show()
from keras.models import Sequential

from keras.layers import Dense, LSTM, Flatten, Activation



model = Sequential()

#hidden layers is Ni + No * (2/3) -> 187 + 5 *(2/3) = 128

model.add(LSTM(128, input_shape=(187, 1), dropout=0.2, recurrent_dropout=0.2,return_sequences=True))

model.add(Flatten())

model.add(Dense(5, activation='softmax')) #output of 5 potential encodings

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



epochs = 15

batch_size = 1



history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)
Y_pred = model.predict(X_valid)



#Converting predictions to label

pred = list()

for i in range(len(Y_pred)):

    pred.append(np.argmax(Y_pred[i]))

#Converting one hot encoded test label to label

test = list()

for i in range(len(Y_valid)):

    test.append(np.argmax(Y_valid[i]))
from sklearn.metrics import accuracy_score

a = accuracy_score(pred,test)

print('Accuracy is:', a*100)
import random



rand_ind = random.randint(0,802)

print(rand_ind)

plt.plot(X_valid[rand_ind,:,:])



class_dict = {0: "normal", 1: "Supraventricular Premature Beat", 2: "Premature ventricular contraction", 3:"Fusion of ventricular and normal beat", 4:"Unclassifiable beat"}



print("Predicted class: ", class_dict[pred[rand_ind]])

print("Actual class: ", class_dict[test[rand_ind]])
# Math

import math

import numpy as np



# Tools

def tabulate(x, y, f):

    """Return a table of f(x, y). Useful for the Gram-like operations."""

    return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))



def cos_sum(a, b):

    """To work with tabulate."""

    return(math.cos(a+b))



def sin_diff(a, b):

    """To work with tabulate."""

    return(math.sin(a-b))



def create_time_serie(size, time):

    """Generate a time serie of length size and dynamic with respect to time."""

    # Generating time-series

    support = np.arange(0, size)

    serie = np.cos(support + float(time))

    return(t, serie)
# Math

import math

import numpy as np



class GASF:



    def __init__(self):

        pass

    def transform(self, serie):

        """Compute the Gramian Angular Field of an image"""

        # Min-Max scaling

        min_ = np.amin(serie)

        max_ = np.amax(serie)

        scaled_serie = (2*serie - max_ - min_)/(max_ - min_)



        # Floating point inaccuracy!

        scaled_serie = np.where(scaled_serie >= 1., 1., scaled_serie)

        scaled_serie = np.where(scaled_serie <= -1., -1., scaled_serie)



        # Polar encoding

        phi = np.arccos(scaled_serie)

        # Note! The computation of r is not necessary

        r = np.linspace(0, 1, len(scaled_serie))



        # GAF Computation (every term of the matrix)

        gaf = tabulate(phi, phi, cos_sum)



        return(gaf, phi, r, scaled_serie)

    

class GADF:



    def __init__(self):

        pass

    def transform(self, serie):

        """Compute the Gramian Angular Field of an image"""

        # Min-Max scaling

        min_ = np.amin(serie)

        max_ = np.amax(serie)

        scaled_serie = (2*serie - max_ - min_)/(max_ - min_)



        # Floating point inaccuracy!

        scaled_serie = np.where(scaled_serie >= 1., 1., scaled_serie)

        scaled_serie = np.where(scaled_serie <= -1., -1., scaled_serie)



        # Polar encoding

        phi = np.arccos(scaled_serie)

        # Note! The computation of r is not necessary

        r = np.linspace(0, 1, len(scaled_serie))



        # GAF Computation (every term of the matrix)

        gaf = tabulate(phi, phi, sin_diff)



        return(gaf, phi, r, scaled_serie)
gasf = GASF()

gadf = GADF()
x_train2 = X_train.reshape(2403,187)

x_valid2 = X_valid.reshape(802,187)
x_train_gasf_images = np.zeros((2403,187,187))

counter = 0

for i in x_train2:

    img = gasf.transform(i)

    x_train_gasf_images[counter] = img[0]

    counter = counter + 1
x_train_gadf_images = np.zeros((2403,187,187))

counter = 0

for i in x_train2:

    img = gadf.transform(i)

    x_train_gadf_images[counter] = img[0]

    counter = counter + 1
x_valid_gasf_images = np.zeros((802,187,187))

counter2 = 0

for i in x_valid2:

    img = gasf.transform(i)

    x_valid_gasf_images[counter2] = img[0]

    counter2 = counter2 + 1

x_valid_gadf_images = np.zeros((802,187,187))

counter2 = 0

for i in x_valid2:

    img = gadf.transform(i)

    x_valid_gadf_images[counter2] = img[0]

    counter2 = counter2 + 1
print(x_train_gasf_images.shape)

print(x_valid_gasf_images.shape)

print(x_train_gadf_images.shape)

print(x_valid_gadf_images.shape)
!pip install git+https://github.com/johannfaouzi/pyts.git
import matplotlib.pyplot as plt

from pyts.image import MarkovTransitionField

from pyts.datasets import load_gunpoint
print(np.transpose(x_train2[0].reshape(-1,1)).shape)

print(X_train[0,:,:].shape)
import warnings

warnings.filterwarnings("ignore")



mtf = MarkovTransitionField(image_size=187)

x_train_mtf_images = np.zeros((2403,187,187))

counter = 0

for i in x_train2:

    img = mtf.fit_transform(np.transpose(i.reshape(-1,1)))

    x_train_mtf_images[counter] = img[0]

    counter = counter + 1
import warnings

warnings.filterwarnings("ignore")



mtf = MarkovTransitionField(image_size=187)

x_valid_mtf_images = np.zeros((802,187,187))

counter = 0

for i in x_valid2:

    img = mtf.fit_transform(np.transpose(i.reshape(-1,1)))

    x_valid_mtf_images[counter] = img[0]

    counter = counter + 1
print(Y_train[512])

print(Y_train[67])

print(Y_train[55])

print(Y_train[9])

print(Y_train[77])
f, axs = plt.subplots(5,4,figsize=(10,10))





plt.subplot(5,4,1)

plt.ylabel("Normal")

plt.plot(X_train[512])

plt.subplot(5,4,2)

plt.imshow(x_train_gasf_images[512])

plt.subplot(5,4,3)

plt.imshow(x_train_gadf_images[512])

plt.subplot(5,4,4)

plt.imshow(x_train_mtf_images[512])



plt.subplot(5,4,5)

plt.ylabel("Supraventricular Premature")

plt.plot(X_train[67])

plt.subplot(5,4,6)

plt.imshow(x_train_gasf_images[67])

plt.subplot(5,4,7)

plt.imshow(x_train_gadf_images[67])

plt.subplot(5,4,8)

plt.imshow(x_train_mtf_images[67])



plt.subplot(5,4,9)

plt.ylabel("Premature VC")

plt.plot(X_train[55])

plt.subplot(5,4,10)

plt.imshow(x_train_gasf_images[55])

plt.subplot(5,4,11)

plt.imshow(x_train_gadf_images[55])

plt.subplot(5,4,12)

plt.imshow(x_train_mtf_images[55])



plt.subplot(5,4,13)

plt.ylabel("Fusion")

plt.plot(X_train[9])

plt.subplot(5,4,14)

plt.imshow(x_train_gasf_images[9])

plt.subplot(5,4,15)

plt.imshow(x_train_gadf_images[9])

plt.subplot(5,4,16)

plt.imshow(x_train_mtf_images[9])



plt.subplot(5,4,17)

plt.xlabel("Timeseries")

plt.ylabel("Unclassifiable Beat")

plt.plot(X_train[77])

plt.subplot(5,4,18)

plt.xlabel("GASF")

plt.imshow(x_train_gasf_images[77])

plt.subplot(5,4,19)

plt.xlabel("GADF")

plt.imshow(x_train_gadf_images[77])

plt.subplot(5,4,20)

plt.xlabel("MTF")

plt.imshow(x_train_mtf_images[77])



x_train_new = np.concatenate((np.expand_dims(x_train_gasf_images, axis=3),np.expand_dims(x_train_gadf_images, axis=3), np.expand_dims(x_train_mtf_images, axis=3)), axis=3)
print(x_train_gasf_images.shape)

print(x_train_gadf_images.shape)

print(x_train_mtf_images.shape)

print(x_train_new.shape)
x_valid_new = np.concatenate((np.expand_dims(x_valid_gasf_images, axis=3),np.expand_dims(x_valid_gadf_images, axis=3), np.expand_dims(x_valid_mtf_images, axis=3)), axis=3)
print(x_valid_gasf_images.shape)

print(x_valid_gadf_images.shape)

print(x_valid_mtf_images.shape)

print(x_valid_new.shape)
from keras.applications.resnet50 import ResNet50

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense
from keras.layers import *

from keras.models import Sequential

from keras.applications.resnet50 import ResNet50



CLASS_COUNT = 5



base_model = ResNet50(

    weights='imagenet',

    include_top=False, 

    input_shape=(187, 187, 3), 

    pooling='avg',

)

base_model.trainable = False



model = Sequential([

  base_model,

  Dense(CLASS_COUNT, activation='softmax'),

])



#compile model using accuracy to measure model performance

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("training")

print(x_train_new.shape)

print(Y_train.shape)

print("validation")

print(x_valid_new.shape)

print(Y_valid.shape)
#train the model

history = model.fit(x_train_new, Y_train, validation_data=(x_valid_new, Y_valid), epochs=10)
plot_model_history(history)
from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D

#create model

model = Sequential()

#add model layers

model.add(Conv2D(64, kernel_size=3, activation='softmax', input_shape=(187,187,3)))

model.add(Dropout(0.2))

model.add(Conv2D(32, kernel_size=3, activation='softmax'))

model.add(Dropout(0.2))

model.add(MaxPool2D(pool_size = (2, 2)))





model.add(Flatten())

model.add(Dense(5, activation='softmax'))



#compile model using accuracy to measure model performance

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train the model

history = model.fit(x_train_new, Y_train, validation_data=(x_valid_new, Y_valid), epochs=15)
plot_model_history(history)