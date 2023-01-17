# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from PIL import Image

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
parasitized = os.listdir("../input/cell_images/cell_images/Parasitized")

uninfected = os.listdir("../input/cell_images/cell_images/Uninfected")
parasitized.remove("Thumbs.db")

uninfected.remove("Thumbs.db")
parasitized_images = []

for p in parasitized:

    img = Image.open("../input/cell_images/cell_images/Parasitized/"+p)

    img = img.resize((50,50))

    parasitized_images.append(img)



uninfected_images = []

for u in uninfected:

    img = Image.open("../input/cell_images/cell_images/Uninfected/"+u)

    img = img.resize((50,50))

    uninfected_images.append(img)
rndm = np.random.randint(len(parasitized_images)-1,size = 10)

plt.figure(1, figsize=(15,7))

for i in range(1,11):

        plt.subplot(2,5,i)

        if i < 6:

            plt.imshow(parasitized_images[rndm[i-1]])

            plt.axis("off")

            plt.title("Parasitized")

        else:

            plt.imshow(uninfected_images[rndm[i-1]])

            plt.axis("off")

            plt.title("Uninfected")
x_array = np.empty((len(parasitized_images)+len(uninfected_images), 50, 50, 3))

x_array = x_array.astype(int)
index = 0

for i in range(x_array.shape[0]):

    if i < len(parasitized_images):

        x_array[i] = np.array(parasitized_images[i])

    else:

        x_array[i] = np.array(uninfected_images[index])

        index += 1
y_array = np.append(np.ones(len(parasitized_images)), np.zeros(len(uninfected_images)))
from keras.utils.np_utils import to_categorical

y_array = to_categorical(y_array, num_classes = 2)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, random_state = 42, test_size = 0.1)

print("x_train shape: ",x_train.shape)

print("x_test shape: ",x_test.shape)

print("y_train shape: ",y_train.shape)

print("y_test shape: ",y_test.shape)
plt.imshow(x_train[1991])

plt.axis("off")

plt.title("Sample")

plt.show()
from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu', input_shape = (50,50,3)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(BatchNormalization(axis = -1))

model.add(Dropout(0.2))



model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(BatchNormalization(axis = -1))

model.add(Dropout(0.2))



model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(BatchNormalization(axis = -1))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(512, activation = "relu"))

model.add(BatchNormalization(axis = -1))

model.add(Dropout(0.5))

model.add(Dense(2, activation = "softmax"))
model.compile(optimizer = "Adam" , loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 20

batch_size = 32
datagen = ImageDataGenerator(

        featurewise_center=False,

        samplewise_center=False,

        featurewise_std_normalization=False,

        samplewise_std_normalization=False,

        zca_whitening=False,

        rotation_range=0.5,

        zoom_range = 0.5,

        width_shift_range=0.5,

        height_shift_range=0.5,

        horizontal_flip=False,

        vertical_flip=False)



datagen.fit(x_train)
history = model.fit(x_train,y_train,epochs=epochs, batch_size=batch_size)
plt.plot(history.history['acc'], color='r', label="accuracies")

plt.title("Train Accuracies")

plt.xlabel("Number of Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
print("Test accuracy: {} %".format(round(model.evaluate(x_test,y_test)[1]*100,2)))