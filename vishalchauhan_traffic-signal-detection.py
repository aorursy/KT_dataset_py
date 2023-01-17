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
# import the necessary packages

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Activation

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Dense



class TrafficSignNet:

	@staticmethod

	def build(width, height, depth, classes):

		# initialize the model along with the input shape to be

		# "channels last" and the channels dimension itself

		model = Sequential()

		inputShape = (height, width, depth)

		chanDim = -1



		# CONV => RELU => BN => POOL

		model.add(Conv2D(8, (5, 5), padding="same",

			input_shape=inputShape))

		model.add(Activation("relu"))

		model.add(BatchNormalization(axis=chanDim))

		model.add(MaxPooling2D(pool_size=(2, 2)))



		# first set of (CONV => RELU => CONV => RELU) * 2 => POOL

		model.add(Conv2D(16, (3, 3), padding="same"))

		model.add(Activation("relu"))

		model.add(BatchNormalization(axis=chanDim))

		model.add(Conv2D(16, (3, 3), padding="same"))

		model.add(Activation("relu"))

		model.add(BatchNormalization(axis=chanDim))

		model.add(MaxPooling2D(pool_size=(2, 2)))



		# second set of (CONV => RELU => CONV => RELU) * 2 => POOL

		model.add(Conv2D(32, (3, 3), padding="same"))

		model.add(Activation("relu"))

		model.add(BatchNormalization(axis=chanDim))

		model.add(Conv2D(32, (3, 3), padding="same"))

		model.add(Activation("relu"))

		model.add(BatchNormalization(axis=chanDim))

		model.add(MaxPooling2D(pool_size=(2, 2)))



		# first set of FC => RELU layers

		model.add(Flatten())

		model.add(Dense(128))

		model.add(Activation("relu"))

		model.add(BatchNormalization())

		model.add(Dropout(0.5))



		# second set of FC => RELU layers

		model.add(Flatten())

		model.add(Dense(128))

		model.add(Activation("relu"))

		model.add(BatchNormalization())

		model.add(Dropout(0.5))



		# softmax classifier

		model.add(Dense(classes))

		model.add(Activation("softmax"))



		# return the constructed network architecture

		return model
import pandas as pd

Meta = pd.read_csv("../input/gtsrb-german-traffic-sign/Meta.csv")

Test = pd.read_csv("../input/gtsrb-german-traffic-sign/Test.csv")

Train = pd.read_csv("../input/gtsrb-german-traffic-sign/Train.csv")

signnames = pd.read_csv("../input/signnames/signnames.csv")
# USAGE

# python train.py --dataset gtsrb-german-traffic-sign --model output/trafficsignnet.model --plot output/plot.png



# set the matplotlib backend so figures can be saved in the background

import matplotlib

matplotlib.use("Agg")



# import the necessary packages

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report

from skimage import transform

from skimage import exposure

from skimage import io

import matplotlib.pyplot as plt

import numpy as np

import argparse

import random

import os



def load_split(basePath, csvPath):

	# initialize the list of data and labels

	data = []

	labels = []



	# load the contents of the CSV file, remove the first line (since

	# it contains the CSV header), and shuffle the rows (otherwise

	# all examples of a particular class will be in sequential order)

	rows = open(csvPath).read().strip().split("\n")[1:]

	random.shuffle(rows)



	# loop over the rows of the CSV file

	for (i, row) in enumerate(rows):

		# check to see if we should show a status update

		if i > 0 and i % 1000 == 0:

			print("[INFO] processed {} total images".format(i))



		# split the row into components and then grab the class ID

		# and image path

		(label, imagePath) = row.strip().split(",")[-2:]



		# derive the full path to the image file and load it

		imagePath = os.path.sep.join([basePath, imagePath])

		image = io.imread(imagePath)



		# resize the image to be 32x32 pixels, ignoring aspect ratio,

		# and then perform Contrast Limited Adaptive Histogram

		# Equalization (CLAHE)

		image = transform.resize(image, (32, 32))

		image = exposure.equalize_adapthist(image, clip_limit=0.1)



		# update the list of data and labels, respectively

		data.append(image)

		labels.append(int(label))



	# convert the data and labels to NumPy arrays

	data = np.array(data)

	labels = np.array(labels)



	# return a tuple of the data and labels

	return (data, labels)





dataset = "../input/gtsrb-german-traffic-sign"

model_path = "../working/trafficsignnet.model"

plot = "../working/plot.png"



# initialize the number of epochs to train for, base learning rate,

# and batch size

NUM_EPOCHS = 30

INIT_LR = 1e-3

BS = 64



# load the label names

labelNames = open("../input/signnames/signnames.csv").read().strip().split("\n")[1:]

labelNames = [l.split(",")[1] for l in labelNames]



# derive the path to the training and testing CSV files

trainPath = os.path.sep.join([dataset, "Train.csv"])

testPath = os.path.sep.join([dataset, "Test.csv"])



# load the training and testing data

print("[INFO] loading training and testing data...")

(trainX, trainY) = load_split(dataset, trainPath)

(testX, testY) = load_split(dataset, testPath)



# scale data to the range of [0, 1]

trainX = trainX.astype("float32") / 255.0

testX = testX.astype("float32") / 255.0



# one-hot encode the training and testing labels

numLabels = len(np.unique(trainY))

trainY = to_categorical(trainY, numLabels)

testY = to_categorical(testY, numLabels)



# account for skew in the labeled data

classTotals = trainY.sum(axis=0)

classWeight = classTotals.max() / classTotals



# construct the image generator for data augmentation

aug = ImageDataGenerator(

	rotation_range=10,

	zoom_range=0.15,

	width_shift_range=0.1,

	height_shift_range=0.1,

	shear_range=0.15,

	horizontal_flip=False,

	vertical_flip=False,

	fill_mode="nearest")



# initialize the optimizer and model

print("[INFO] compiling model...")

opt = Adam(lr=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5))

model = TrafficSignNet.build(width=32, height=32, depth=3,

	classes=numLabels)

model.compile(loss="categorical_crossentropy", optimizer=opt,

	metrics=["accuracy"])



# compile the model and train the network

print("[INFO] training network...")

H = model.fit_generator(

	aug.flow(trainX, trainY, batch_size=BS),

	validation_data=(testX, testY),

	steps_per_epoch=trainX.shape[0] // BS,

	epochs=NUM_EPOCHS,

	class_weight=classWeight,

	verbose=1)



# evaluate the network

print("[INFO] evaluating network...")

predictions = model.predict(testX, batch_size=BS)

print(classification_report(testY.argmax(axis=1),

	predictions.argmax(axis=1), target_names=labelNames))



# save the network to disk

print("[INFO] serializing network to '{}'...".format(model))

model.save(model_path)



# plot the training loss and accuracy

N = np.arange(0, NUM_EPOCHS)

plt.style.use("ggplot")

plt.figure()

plt.plot(N, H.history["loss"], label="train_loss")

plt.plot(N, H.history["val_loss"], label="val_loss")

plt.plot(N, H.history["accuracy"], label="train_acc")

plt.plot(N, H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy on Dataset")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend(loc="lower left")

plt.savefig(plot)
!pip install imutils
# USAGE

# python predict.py --model output/trafficsignnet.model --images gtsrb-german-traffic-sign/Test --examples examples



# import the necessary packages

from tensorflow.keras.models import load_model

from skimage import transform

from skimage import exposure

from skimage import io

from imutils import paths

import numpy as np

import argparse

import imutils

import random

import cv2

import os





example = "../working"

images_path = "../input/gtsrb-german-traffic-sign/Test" 



# load the traffic sign recognizer model

print("[INFO] loading model...")

model = load_model(model_path)



# load the label names

labelNames = open("../input/signnames/signnames.csv").read().strip().split("\n")[1:]

labelNames = [l.split(",")[1] for l in labelNames]



# grab the paths to the input images, shuffle them, and grab a sample

print("[INFO] predicting...")

print(images_path)

imagePaths = list(paths.list_images(images_path))

random.shuffle(imagePaths)

imagePaths = imagePaths[:25]



# loop over the image paths

for (i, imagePath) in enumerate(imagePaths):

	# load the image, resize it to 32x32 pixels, and then apply

	# Contrast Limited Adaptive Histogram Equalization (CLAHE),

	# just like we did during training

	image = io.imread(imagePath)

	image = transform.resize(image, (32, 32))

	image = exposure.equalize_adapthist(image, clip_limit=0.1)



	# preprocess the image by scaling it to the range [0, 1]

	image = image.astype("float32") / 255.0

	image = np.expand_dims(image, axis=0)



	# make predictions using the traffic sign recognizer CNN

	preds = model.predict(image)

	j = preds.argmax(axis=1)[0]

	label = labelNames[j]



	# load the image using OpenCV, resize it, and draw the label

	# on it

	image = cv2.imread(imagePath)

	image = imutils.resize(image, width=128)

	cv2.putText(image, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,

		0.45, (0, 0, 255), 2)



	# save the image to disk

	p = os.path.sep.join([example, "{}.png".format(i)])

	cv2.imwrite(p, image)