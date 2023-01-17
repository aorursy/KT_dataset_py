#Get the input files
import os
import pandas as pd 
import numpy as np
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding


print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Getting to knoe the data
train.info()
print(train.shape)
print(train.describe())
test.info()
print(test.shape)
print(test.describe())
#Restructure the data


#1. Separate Label and Image Pixels
train_label = train["label"]
train_images = train.drop(labels = ["label"], axis = 1)

#2. Normalize the data (Since, 255 is the max value)
train_images = train_images / 255.0
test = test / 255.0

#3. Reshape the data - From scalar to actual image format -?? 
train_images = train_images.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

#4. Label Encoding - Changing classes to 0/1 matrices
train_label = to_categorical(train_label, num_classes = 10)


print(train_images.shape)
print(test.shape)
print(train_label.shape)

#Define model

from keras import models
from keras import layers
from keras.layers.core import  Flatten, Dense, Activation

network = models.Sequential()
network.add(layers.Dense(52, activation='relu', input_shape=(28, 28, 1)))
network.add(Flatten())
network.add(Dense(10, activation='softmax'))

print(network.input_shape)
print(network.output_shape)

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

network.fit(train_images, train_label, epochs=5, batch_size=128)
predictions = network.predict_classes(test)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)
