import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical

train_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/train'

imagePaths = []
for dirname, _, filenames in os.walk(train_path):
    for filename in filenames:
        imagePath = os.path.join(dirname, filename)
        imagePaths.append(imagePath)
image_size = 96
def pre_processing(imagePath) :
    img = cv2.imread(imagePath) 
    img = cv2.resize(img, (image_size,image_size))
    img = img_to_array(img)
    return img
def process_all(imagePaths):
    data = []
    for imagePath in imagePaths:
        img = pre_processing(imagePath)
        data.append(img)
    return data
data = process_all(imagePaths)
def label(imagePaths):
    labels = []
    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        if label == 'NORMAL': labels.append(0)
        elif label == 'PNEUMONIA': labels.append(1)
    return labels
labels = label(imagePaths)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
x_train = data
y_train = to_categorical(labels, num_classes=2)
(x_train, x_val, y_train, y_val) = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                horizontal_flip=True, fill_mode="nearest")

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
EPOCHS = 25
INIT_LR = 1e-3
BS = 32
def our_model(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model.add(Conv2D(32, (3, 3), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        #model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        # return the constructed network architecture
        return model
    
model = our_model(image_size,image_size,3,2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit_generator(aug.flow(x_train, y_train, batch_size = BS), epochs = EPOCHS, validation_data = aug.flow(x_val, y_val))
import matplotlib.pyplot as plt

fig , ax = plt.subplots(1,2)
fig.set_size_inches(20,10)

ax[0].plot( H.history["loss"],'go-', label="train_loss")
ax[0].plot( H.history["val_loss"],'ro-', label="val_loss")
ax[0].plot( H.history["accuracy"], label="train_acc")
ax[0].plot( H.history["val_accuracy"], label="val_acc")
ax[0].set_title("Learning curves for predicting lung ìnfection")
ax[0].legend(loc="lower left")
ax[0].set_xlabel("Epoch #")
ax[0].set_ylabel("Loss/Accuracy")
plt.show()
