import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.utils import np_utils

%matplotlib inline

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')
# each entry is a flattened array for the pixel values of a 28x28 image, with the first 

# column corresponding to the label of the image

train.head(5)
test.head(5)
print(test[:1].shape)

print(train[:1].shape)
# Lets clean this up a bit better so that the pixel values and the labels are separated

# note that this converts X_train to a np array

X_train=train.iloc[:, 1:].values

y_train=train.iloc[:, 0].values

X_test = test.values



print("Size of training data: {}".format(X_train.shape))

print("Size of test data {}".format(test.shape))

print("Size of a single entry in X_train {}".format(X_train[:1].shape))

#print(X_test.describe)

# Next, we will normalize the values to be from 0 to 1

# normalize pixel values from 0-255 to 0-1

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')



X_train = X_train / 255

X_test = X_test / 255
X_train.shape

X_test.shape

# one hot encode outputs

y_train = np_utils.to_categorical(y_train)

num_classes = y_train.shape[1]

print(num_classes)

print(y_train[1:5])
img_width=28

img_height=28

img_depth=1



plt.figure(figsize=(12,10))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.imshow(X_train[i].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Label {}".format((np.where(y_train[i]==1))))
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout
# set random seed for reproducibility

seed = 7

np.random.seed(seed)
def baseline_model():

    model = Sequential()

    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal',activation='relu'))

    model.add(Dense(num_classes, input_dim=num_pixels, kernel_initializer='normal',activation='softmax'))

    # compile model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
num_pixels = X_train.shape[1]

# num_classes has already been defined
# first build the model

model = baseline_model()



# next, fit the model

model.fit(X_train, y_train, epochs=10, batch_size=200, verbose=2)
y_train[1:10,:]
from keras.models import load_model

model.save('baseline.h5')

model=load_model('baseline.h5')

results=model.predict_classes(X_test)



# results are not one hot encoded (?) so these are just the labels

results = pd.Series(results,name="Label")

#print(results[1:10])
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("baseline_model.csv",index=False)
print(check_output(["ls", "."]).decode("utf8"))
from keras.layers import Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils

from keras import backend as K

K.set_image_dim_ordering('th')
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)

X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)



print("Size of training data: {}".format(X_train.shape))

print("Size of test data {}".format(X_test.shape))
def cnn_model():

    # build model

    model = Sequential()

    # first build input layer, expecting images in pixels, width, height

    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))

    # Next is a pooling layer

    model.add(MaxPooling2D(pool_size=(2, 2)))    

    # Next layer is to randomly exclude 20% of neurons to prevent overfitting

    model.add(Dropout(0.2))

    # Next layer converts 2D matrix data to a vector (flatten)

    model.add(Flatten())

    # Next a fully connected layer with 128 neurons and relu

    model.add(Dense(128, activation='relu'))

    # Finally output layer

    model.add(Dense(num_classes, activation='softmax'))

    # next, compile model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # TODO: should be categorical_crossentropy (as y_train has been one-hot encoded)

    # but throws error on final layer size mismatch. sparse_categorical_crossentropy works with only digits

    # see https://github.com/fchollet/keras/issues/3009

    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
# create model

model = cnn_model()



# lets have a look at the model summary

model.summary()

X_train.reshape((-1, 1))

# fit the model - slow computer.. only try 3 epochs

model.fit(X_train, y_train, epochs=3, batch_size=50, verbose=2)
# now to predict the model's accuracy on the test data set

from keras.models import load_model

model.save('model.h5')

model=load_model('model.h5')

labels=model.predict_classes(X_test)
labels = pd.Series(labels,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),labels],axis = 1)



submission.to_csv("cnn_model.csv",index=False)

print(check_output(["ls", "."]).decode("utf8"))