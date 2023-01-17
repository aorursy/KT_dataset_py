# Restart runtime using it'Runtime' -> 'Restart runtime...'

%tensorflow_version 1.x

import tensorflow as tf

print(tf.__version__)

from google.colab import drive

drive.mount('/content/drive')
import os

os.chdir('/content/drive/My Drive/Dataset')  #change dir

!mkdir invalid  #create a directory named train/

!unzip -q "Invalid.zip" -d invalid/  #unzip data in train/
import os

os.chdir('/content/drive/My Drive/Dataset')  #change dir

!mkdir Retina  #create a directory named train/

!unzip -q "Retina.zip" -d Retina/  #unzip data in train/
from keras.models import Sequential

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dense

from keras import backend as K

import matplotlib

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import img_to_array

from keras.utils import to_categorical

from imutils import paths

import matplotlib.pyplot as plt

import numpy as np

import argparse

import random

import cv2

import os







EPOCHS = 25

INIT_LR = 1e-3

BS = 32
data = []

labels = []

DATADIR = "/content/drive/My Drive/Dataset"



imagePaths = sorted(list(paths.list_images(DATADIR)))

random.seed(42)

random.shuffle(imagePaths)



for imagePath in imagePaths:

	# load the image, pre-process it, and store it in the data list

	image = cv2.imread(imagePath)

	image = cv2.resize(image, (224, 224))

	image = img_to_array(image)

	data.append(image)



	# extract the class label from the image path and update the

	# labels list

	label = imagePath.split(os.path.sep)[-2]

	label = 1 if label == "Retina" else 0

	labels.append(label)



# scale the raw pixel intensities to the range [0, 1]

data = np.array(data, dtype="float") / 255.0

labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data,

	labels, test_size=0.25, random_state=42)
trainY = to_categorical(trainY, num_classes=2)

testY = to_categorical(testY, num_classes=2)
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,

	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,

	horizontal_flip=True, fill_mode="nearest")
width=224

height=224

depth=3

classes=2



inputShape = (height, width, depth)

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)





model = Sequential()

		



		# if we are using "channels first", update the input shape

if K.image_data_format() == "channels_first":

	inputShape = (depth, height, width)



		# first set of CONV => RELU => POOL layers

model.add(Conv2D(20, (5, 5), padding="same",

			input_shape=inputShape))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



		# second set of CONV => RELU => POOL layers

model.add(Conv2D(50, (5, 5), padding="same"))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



		# first (and only) set of FC => RELU layers

model.add(Flatten())

model.add(Dense(500))

model.add(Activation("relu"))



		# softmax classifier

model.add(Dense(classes))

model.add(Activation("softmax"))

  

model.compile(loss="binary_crossentropy", optimizer=opt,

	metrics=["accuracy"])



H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),

	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,

	epochs=EPOCHS, verbose=1)
model.save("validation-version-02-050220.h5")
plt.style.use("ggplot")

plt.figure()

N = EPOCHS

plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")

plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")

plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy on Retina/Invalid")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend(loc="lower left")
from keras.preprocessing.image import img_to_array

from keras.models import load_model

import numpy as np

import argparse

import imutils

import cv2



image = cv2.imread("/content/drive/My Drive/Dataset/circle-cropped.png")

orig = image.copy()





image = cv2.resize(image, (224, 224))

image = image.astype("float") / 255.0

image = img_to_array(image)

image = np.expand_dims(image, axis=0)



model = load_model("/content/drive/My Drive/Dataset/validation-version-02-050220.h5")



# classify the input image

(invalid, Retina) = model.predict(image)[0]





label = "Retina" if Retina > invalid else "Invalid"

proba = Retina if Retina > Retina else Retina

label = "{}: {:.2f}%".format(label, proba * 100)



# draw the label on the image

output = imutils.resize(orig, width=400)

cv2.putText(output, label, (25, 25),  cv2.FONT_HERSHEY_SIMPLEX,

	1, (0, 255, 0), 2)



# show the output image

plt.imshow(output)

plt.show()

image = cv2.imread("/content/drive/My Drive/Dataset/download (1).jfif")

orig = image.copy()





image = cv2.resize(image, (28, 28))

image = image.astype("float") / 255.0

image = img_to_array(image)

image = np.expand_dims(image, axis=0)



model = load_model("/content/drive/My Drive/Dataset/validation.h5")



# classify the input image

(invalid, Retina) = model.predict(image)[0]





label = "Retina" if Retina > invalid else "Invalid"

proba = Retina if Retina > Retina else Retina

label = "{}: {:.2f}%".format(label, proba * 100)



# draw the label on the image

output = imutils.resize(orig, width=400)

cv2.putText(output, label, (25, 25),  cv2.FONT_HERSHEY_SIMPLEX,

	1, (0, 255, 0), 2)



# show the output image

plt.imshow(output)

plt.show()