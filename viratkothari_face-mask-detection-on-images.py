from IPython.display import Image

Image("../input/face-detector/header-img.jpg")
# Import Libraries



!pip install imutils



from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.layers import AveragePooling2D

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Input

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from imutils import paths

import matplotlib.pyplot as plt

import numpy as np

import argparse

import os
# Initializing the variables the initial learning rate, number of epochs to train for, and batch size



INIT_LR = 1e-4 # Initial learning rate

EPOCHS = 20 # Number of epochs

BS = 32 # Batch size



imagePaths = list(paths.list_images("../input/facemaskimages"))

data = []

labels = []
# Loading images in the data[] and Labels[] variable



# grab the list of images in our dataset directory, then initialize

# the list of data (i.e., images) and class images



print("[INFO] loading images...")

# print(len(imagePaths));



# loop over the image paths

for imagePath in imagePaths:

    # extract the class label from the filename

    label = imagePath.split(os.path.sep)[-2]

    # load the input image (224x224) and preprocess it

    image = load_img(imagePath, target_size=(224, 224))

    image = img_to_array(image)

    image = preprocess_input(image)

    # update the data and labels lists, respectively

    data.append(image)

    labels.append(label)

# convert the data and labels to NumPy arrays

data = np.array(data, dtype="float32")

# print(data)

labels = np.array(labels)

# print(labels)
# perform one-hot encoding on the labels

lb = LabelBinarizer()

labels = lb.fit_transform(labels)

labels = to_categorical(labels)

# partition the data into training and testing splits using 80% of

# the data for training and the remaining 20% for testing

(trainX, testX, trainY, testY) = train_test_split(data, labels,

    test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data augmentation

aug = ImageDataGenerator(

    rotation_range=20,

    zoom_range=0.15,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.15,

    horizontal_flip=True,

    fill_mode="nearest")



# print(trainX)

# print(testX)

# print(trainY)

# print(testY)
# load the MobileNetV2 network, ensuring the head FC layer sets are

# left off

baseModel = MobileNetV2(weights="imagenet", include_top=False,

    input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the

# the base model

headModel = baseModel.output

headModel = AveragePooling2D(pool_size=(7, 7))(headModel)

headModel = Flatten(name="flatten")(headModel)

headModel = Dense(128, activation="relu")(headModel)

headModel = Dropout(0.5)(headModel)

headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become

# the actual model we will train)

model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will

# *not* be updated during the first training process

for layer in baseModel.layers:

    layer.trainable = False
# compile our model

print("[INFO] compiling model...")

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="binary_crossentropy", optimizer=opt,

    metrics=["accuracy"])

# train the head of the network

print("[INFO] training head...")

H = model.fit(

    aug.flow(trainX, trainY, batch_size=BS),

    steps_per_epoch=len(trainX) // BS,

    validation_data=(testX, testY),

    validation_steps=len(testX) // BS,

    epochs=EPOCHS)
# make predictions on the testing set

print("[INFO] evaluating network...")

predIdxs = model.predict(testX, batch_size=BS)

# print(predIdxs)





# for each image in the testing set we need to find the index of the

# label with corresponding largest predicted probability

predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report

print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# serialize the model to disk

# print("[INFO] saving mask detector model...")

# model.save("./mask_detector.model", save_format="h5")
# plot the training loss and accuracy

N = EPOCHS

plt.style.use("ggplot")

plt.figure()

plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")

plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")

plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend(loc="lower left")

plt.savefig("./plot.png")
# import the necessary packages

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.models import load_model

import numpy as np

import argparse

import cv2

import os
# load our serialized face detector model from disk

print("[INFO] loading face detector model...")

prototxtPath = "../input/face-detector/deploy.prototxt"

weightsPath = "../input/face-detector/res10_300x300_ssd_iter_140000.caffemodel"



print(prototxtPath)

print(weightsPath)



net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk

# print("[INFO] loading face mask detector model...")

# model = load_model(r"./mask_detector.model")
# Function load the input image from disk, clone it, and grab the image spatial

# dimensions



def plot_image(imagepath):

#     print(imagepath)

    image = cv2.imread(imagepath)



    orig = image.copy()

    (h, w) = image.shape[:2]

    # construct a blob from the image

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),

        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections

#     print("[INFO] computing face detections...")

    net.setInput(blob)

    detections = net.forward()

    

    # loop over the detections

    for i in range(0, detections.shape[2]):

        # extract the confidence (i.e., probability) associated with

        # the detection

        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is

        # greater than the minimum confidence

        if confidence > 0.5:

            # compute the (x, y)-coordinates of the bounding box for

            # the object

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of

            # the frame

            (startX, startY) = (max(0, startX), max(0, startY))

            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel

            # ordering, resize it to 224x224, and preprocess it

            face = image[startY:endY, startX:endX]

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            face = cv2.resize(face, (224, 224))

            face = img_to_array(face)

            face = preprocess_input(face)

            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face

            # has a mask or not

            (mask, withoutMask) = model.predict(face)[0]

            # determine the class label and color we'll use to draw

            # the bounding box and text

            label = "Mask" if mask > withoutMask else "No Mask"

            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label

#             print("{} {:.2f}%".format("", max(mask, withoutMask) * 100))

            label = "{}:{:.2f}%".format(label, max(mask, withoutMask) * 100)

#             print(label)

            # display the label and bounding box rectangle on the output

            # frame

            cv2.putText(image, label, (startX, startY - 10),

                cv2.FONT_HERSHEY_SIMPLEX, 3, color, 10)

            cv2.rectangle(image, (startX, startY), (endX, endY), color, 20)



            # show the output image

    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGBA)

    plt.grid(False)

    plt.axis('off')

    plt.imshow(image)
# Creating data for test images



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

TestImagePath = []



import os

for dirname, _, filenames in os.walk('../input/testimages'):

    for filename in filenames:

        TestImagePath.append(os.path.join(dirname, filename))

#         print(os.path.join(dirname, filename))
# verification of several images

import matplotlib.pyplot as plt

num_rows=4

num_cols=4

num_images=num_rows*num_cols



plt.figure(figsize=(2*2*num_cols,3*num_rows))

i=1

for i in range(num_images):

    plt.subplot(num_rows, 2*num_cols, 2*i+1)

    plot_image(TestImagePath[i])

    

plt.show()
print("Notebook completd!")