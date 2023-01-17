# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from tensorflow.python.client import device_lib

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split # sklearn, train_test_split

import matplotlib.pyplot as plt # matplotlib, plt

from keras.utils.np_utils import to_categorical # categorical - one-hot-encoding



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# CNN libraries

from sklearn.metrics import confusion_matrix

import itertools

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator



# Any results you write to the current directory are saved as output.
def get_available_gpus():

    local_device_protos = device_lib.list_local_devices()

    print(local_device_protos)

    return [x.name for x in local_device_protos if x.device_type == 'GPU']



get_available_gpus()
# import datasets 

print(os.listdir("../input"))

directory_train = "../input/train.csv"

directory_test = "../input/test.csv"
test_data = pd.read_csv(directory_test)

print(test_data.shape)

test_data.head(5)
train_data = pd.read_csv(directory_train)

print(train_data.shape)

train_data.head(5)
## Formatting the Labels

def format_label():

    return [train_data["label"], 

            train_data.drop(labels = ["label"],axis = 1)

           ]



# call format_label

y_train, x_train = format_label()

print(x_train.head(5), y_train.head(5))
# Show some samples

# x_train dataset 

def show_figure():

    for index in range(0, 9):

        fig = x_train.iloc[index].as_matrix()

        fig = fig.reshape((28,28))

        plt.subplot(340 + (index + 1))

        plt.imshow(fig)

        plt.title(y_train[index])

        

# call show_figure      

show_figure()

def normalization():

    return [

            x_train / 255.0,

            test_data / 255.0

           ]

def reshape():

    return [

            x_train.values.reshape(-1,28, 28,1),

            test_data.values.reshape(-1,28,28,1)

           ]

def tensorflow_encoding():

    return to_categorical(y_train, num_classes = 10)



# call normalization

x_train, test_data = normalization()

print('x_train:',x_train.shape)

print('test:',test_data.shape)



# call reshape

print("------ Reshape ---------------")

x_train, test_data = reshape()

print('x_train:',x_train.shape)

print('test:',test_data.shape)



# call tensorflow label encoding

y_train = tensorflow_encoding()

print(y_train)
def separating_data():

    return train_test_split(x_train, 

                            y_train, 

                            test_size = 0.1, 

                            random_state = 42

                           )



x_train, x_validation, y_train, y_validation = separating_data()
activations = ['relu', 'softmax'] 

#Change the number of epochs for other results

# This may be interesting !

max_epochs = 2

model = Sequential()



model.add(Conv2D(filters = 32, 

                 kernel_size = (5,5),

                 padding = 'Same', 

                 activation = activations[0], 

                 input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, 

                 kernel_size = (5,5),

                 padding = 'Same', 

                 activation = activations[0]))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, 

                 kernel_size = (3,3),

                 padding = 'Same', 

                 activation = activations[0]))

model.add(Conv2D(filters = 64, 

                 kernel_size = (3,3),

                 padding = 'Same', 

                 activation = activations[0]))



model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = activations[0]))

model.add(Dropout(0.5))

model.add(Dense(10, activation = activations[1]))



model.compile(loss='categorical_crossentropy', 

              optimizer='adam', 

              metrics=["accuracy"])



generator = ImageDataGenerator()



batches = generator.flow(x_train, 

                         y_train, 

                         batch_size=64)

batches_value = generator.flow(x_validation, 

                               y_validation, 

                               batch_size=64)



history = model.fit_generator(generator=batches, 

                              steps_per_epoch=batches.n, 

                              epochs=max_epochs, 

                              validation_data=batches_value, 

                              validation_steps=batches_value.n)
def show_loss():

    plt.plot(history.history['loss'], 

             color='red', 

             label='Validation Loss')

    plt.title("My Test Loss")

    plt.xlabel("Epochs")

    plt.ylabel("Loss")

    plt.legend()

    plt.show()

    

def show_acc_epochs():

    plt.plot(history.history['acc'],

             color='blue',

             label='Acc')

    plt.title("My Test Accuracy")

    plt.xlabel("Epochs")

    plt.ylabel("Accuracy Value")

    plt.legend()

    plt.show()

    

# call plot loss

show_loss()

# call plt acc_epochs

show_acc_epochs()
# predict results and submit

results = model.predict(test_data)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),

                                  name = "ImageId"),

                        results],axis = 1)

submission.to_csv("submission.csv",index=False)