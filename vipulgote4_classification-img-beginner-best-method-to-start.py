import tensorflow as tf

import keras

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import cv2

import os

import glob

import pickle
import random 

from tensorflow.keras.layers import Conv2D,Dense,MaxPooling2D,Dropout,Flatten,Activation

from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
CATAGORIES=['Bishop','King','Knight','Pawn','Queen','Rook']

DATADIR=(r'../input/chessman-image-dataset/chess')

IMG_SIZE=150

training_data=[]
def create_training_data():

    for catagories in CATAGORIES:

        path=os.path.join(DATADIR,catagories)

        class_num=CATAGORIES.index(catagories)

        for img in os.listdir(path):

            try:

                img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))

                training_data.append([new_array,class_num])

            except Exception as e:

                pass
import os

print(os.listdir("../input"))
create_training_data()
random.shuffle(training_data)
X=[]

y=[]
for feature,label in training_data:

    X.append(feature)

    y.append(label)
X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
X = X/255.0
y=np.array(y)
(trainX, testX, trainY, testY) = train_test_split(X, y,

	test_size=0.20, random_state=42)
model = Sequential()

# 3 convolutional layers

model.add(Conv2D(32, (3, 3), input_shape = trainX.shape[1:]))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3)))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3)))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



# 2 hidden layers

model.add(Flatten())

model.add(Dense(128))

model.add(Activation("relu"))

model.add(Dropout(0.25))



model.add(Dense(128))

model.add(Activation("relu"))



# The output layer with 13 neurons, for 13 classes

model.add(Dense(6))

model.add(Activation("softmax"))
model.summary()
model.compile(loss='sparse_categorical_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])
datagen = ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
#history = model.fit_generator(datagen.flow(trainX, trainY, batch_size=32), epochs=3,steps_per_epoch=len(trainX) / 32)
history=model.fit(trainX, trainY, batch_size=32, epochs=20, verbose=1, validation_data=(testX, testY),callbacks=[callback])
plt.figure(1)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model_Accuracy')

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.legend(['train', 'validation'], loc='upper left')
test_loss, test_acc = model.evaluate(testX,testY)

print(test_acc)
def prepare(file):

    IMG_SIZE = 150

    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

image = testX[1] #your image path

prediction = model.predict([image])

prediction = list(prediction[0])

print(CATEGORIES[prediction.index(max(prediction))])