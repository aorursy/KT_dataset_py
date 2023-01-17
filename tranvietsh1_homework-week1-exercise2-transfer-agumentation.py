# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input/dogandcat/dogandcat/train/dog"))



# Any results you write to the current directory are saved as output.
!pip3 install --upgrade imutils
import numpy as np

from skimage import color, exposure, transform

import cv2

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from imutils import paths

from keras.applications import VGG16

from keras.applications import imagenet_utils

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import SGD

from keras.optimizers import RMSprop

from keras.applications import VGG16

from keras.layers import Input

from keras.models import Model

from keras.layers.core import Dense

from keras.layers.core import Dropout

from keras.layers.core import Flatten

import numpy as np

import random

import os



NUM_CLASSES = 2

IMG_SIZE = 32
def preprocess_img(img):

    # Histogram normalization in v channel

    hsv = color.rgb2hsv(img)

    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])

    img = color.hsv2rgb(hsv)



    # central square crop

    min_side = min(img.shape[:-1])

    centre = img.shape[0] // 2, img.shape[1] // 2

    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,

              centre[1] - min_side // 2:centre[1] + min_side // 2,

              :]



    # rescale to standard size

    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))



    # roll color axis to axis 0

    np.rollaxis(img, 1) 



    return img
import os

import glob



#get name lable

def get_class(img_path):

    return img_path.split('/')[-2]



#root_dir input

root_dir = "../input/dogandcat/dogandcat/train"

imgs = []

labels = []

countDog = 0

countCat = 0



#List all root dir image

all_img_paths = glob.glob(os.path.join(root_dir, '*/*.jpg'))



#swap random image

np.random.shuffle(all_img_paths)



#plot image

for i in range(0,9):

    imgPlot = cv2.imread(all_img_paths[i])

    ax = plt.imshow(imgPlot)

    

#valid input to array  

for img_path in all_img_paths:

    img = preprocess_img(cv2.imread(img_path))

    nameLabel = get_class(img_path)

    if nameLabel =="cat":

        label = 0

        countCat +=1

    else:

        label=1

        countDog +=1

    imgs.append(img)

    labels.append(label)

    #print(img_path)



X = np.array(imgs, dtype='float32')

print(X.shape)

# Make one hot targets

Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

print(Y.shape)

print("Number images train of Cat: " + str(countCat))

print("Number images train of Dog: " + str(countDog))
# Load model VGG 16 của ImageNet dataset, include_top=False để bỏ phần Fully connected layer ở cuối.

baseModel = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))



# Xây thêm các layer

# Lấy output của ConvNet trong VGG16

fcHead = baseModel.output



# Flatten trước khi dùng FCs

fcHead = Flatten(name='flatten')(fcHead)



# Thêm FC

fcHead = Dense(256, activation='relu')(fcHead)

fcHead = Dropout(0.5)(fcHead)



# Output layer với softmax activation

fcHead = Dense(NUM_CLASSES, activation='softmax')(fcHead)



# Xây dựng model bằng việc nối ConvNet của VGG16 và fcHead

model = model = Model(inputs=baseModel.input, outputs=fcHead)
# augmentation cho training data

aug_train = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, 

                         zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

# augementation cho test

aug_test= ImageDataGenerator(rescale=1./255)
root_dir = '../input/dogandcat/dogandcat/test'

imgtest = []

labeltest = []





all_img_paths = glob.glob(os.path.join(root_dir, '*/*.jpg'))

np.random.shuffle(all_img_paths)

for img_path in all_img_paths:

    img = preprocess_img(cv2.imread(img_path))

    nameLabel = get_class(img_path)

    #print(label)

    if nameLabel =="cat":

        label = 0

    else:

        label = 1

    #print(labeltes)

    imgtest.append(img)

    labeltest.append(label)



X_test = np.array(imgtest, dtype='float32')

print(X_test.shape)

# Make one hot targets

labels = np.array(labeltest)

Y_test = np.eye(NUM_CLASSES, dtype='uint8')[labeltest]

print(Y_test.shape)
# freeze VGG model

for layer in baseModel.layers:

    layer.trainable = False

lr=0.001    

opt = RMSprop(lr)

model.compile(opt, 'categorical_crossentropy', ['accuracy'])

numOfEpoch = 25

H = model.fit_generator(aug_train.flow(X, Y, batch_size=32), 

                        steps_per_epoch=len(X)//32,

                        validation_data=(aug_test.flow(X_test, Y_test, batch_size=32)),

                        validation_steps=len(X_test)//32,

                        epochs=numOfEpoch)
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

def lr_schedule(epoch):

    return lr * (0.1 ** int(epoch / 10))

# unfreeze some last CNN layer:

for layer in baseModel.layers[15:]:

    layer.trainable = True



numOfEpoch = 35

opt = SGD(lr)

model.compile(opt, 'categorical_crossentropy', ['accuracy'])

H = model.fit_generator(aug_train.flow(X, Y, batch_size=32), 

                        steps_per_epoch=len(X)//32,

                        validation_data=(aug_test.flow(X_test, Y_test, batch_size=32)),

                        validation_steps=len(X_test)//32,

                        epochs=numOfEpoch,

                        callbacks=[LearningRateScheduler(lr_schedule),

                        ModelCheckpoint('model5.h5', save_best_only=True)])
from keras.models import load_model

model = load_model('model5.h5')



# predict and evaluate

y_pred = model.predict(X_test)

max_pre = np.argmax(y_pred,axis=1)

print("predict: " ,max_pre)

#print(y_pred.shape)

acc = np.sum(max_pre == labels) / np.size(max_pre)

print("Test accuracy = {}".format(acc))