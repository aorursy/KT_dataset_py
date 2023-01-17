    # This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline



from keras.models import Sequential

from keras.layers import Dense , Dropout , Lambda, Flatten

from keras.optimizers import Adam ,RMSprop

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
print("LOAD TRAIN/TEST DATA: ")



# create the training & test sets, skipping the header row with [1:]

train = pd.read_csv("../input/train.csv")



test_images = (pd.read_csv("../input/test.csv").values).astype('float32')

train_images = (train.ix[:,1:].values).astype('float32')

train_labels = train.ix[:,0].values.astype('int32')

print("train_images: SHAPE: ")

print(train_images.shape)
print("SHOW 3 images:")

#Convert train datset to (num_images, img_rows, img_cols) format 



train_images = train_images.reshape(train_images.shape[0],  28, 28)



for i in range(6, 9):

    plt.subplot(330 + (i+1))

    plt.imshow(train_images[i], cmap=plt.get_cmap('gray'))

    plt.title(train_labels[i]);

print("CHANGE THE SHAPE OF TEST DATA AS WELL:")

train_images = train_images.reshape((42000, 28 * 28))

print("SHAPE OF TEST IMAGE:")

print(test_images.shape)

print("SHAPE OF TRAIN LABEL SHAPE:")

print(train_labels.shape)

print("TRAIN LABELS:")

print(train_labels)
print("Feature Standardization: make pixel values between 0 - 1: ")

train_images = train_images / 255

test_images = test_images / 255
print("ONE HOT ENCODING OF LABELS: ")

from keras.utils.np_utils import to_categorical

train_labels = to_categorical(train_labels)

num_classes = train_labels.shape[1]

print("NUMBER OF CLASSES:")

print(num_classes)
print("DESIGNING NETWORK:")

# fix random seed for reproducibility

seed = 43

np.random.seed(seed)



from keras.models import Sequential

from keras.layers import Dense , Dropout



model=Sequential()

model.add(Dense(32,activation='relu',input_dim=(28 * 28)))

model.add(Dense(16,activation='relu'))

model.add(Dense(10,activation='softmax'))

print("COMPILE NETWORK: ")

print("WHAT WE NEED: 1. A loss function: to measure how good the network is 2. An optimizer: to update network as it sees more data and reduce loss value 3. Metrics: to monitor performance of network")



from keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),

 loss='categorical_crossentropy',

 metrics=['accuracy'])

print("TRAINING ")

history=model.fit(train_images, train_labels, validation_split = 0.05, 

            epochs=25, batch_size=64)
history_dict = history.history

print("history dictionary keys: ")

print(history_dict.keys())
print("GRAPH FOR LOSSES OVER EPOCHS:")

import matplotlib.pyplot as plt

%matplotlib inline

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

# "bo" is for "blue dot"

plt.plot(epochs, loss_values, 'bo')

# b+ is for "blue crosses"

plt.plot(epochs, val_loss_values, 'b+')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.show()
print("GRAPH FOR ACCURACY OVER EPOCHS:")

plt.clf()   # clear figure

acc_values = history_dict['acc']

val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'bo')

plt.plot(epochs, val_acc_values, 'b+')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.show()