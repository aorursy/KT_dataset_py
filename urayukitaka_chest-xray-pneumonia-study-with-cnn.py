# Basic library

import numpy as np 

import pandas as pd 

import os

import glob



# Data preprocessing

import cv2 # Open cv



# Visualization

from matplotlib import pyplot as plt

import seaborn as sns

sns.set()



# Machine learning library

import keras

from keras.models import Sequential, Model, load_model

from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, Activation, Input

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.preprocessing.image import ImageDataGenerator



# Validation

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
print("train",os.listdir("../input/chest-xray-pneumonia/chest_xray/train"))

print("val",os.listdir("../input/chest-xray-pneumonia/chest_xray/train"))

print("test",os.listdir("../input/chest-xray-pneumonia/chest_xray/train"))
# train_data_set

# Normal

train_data_nor = pd.DataFrame({})

train_data_nor["data_id"] = os.listdir("../input/chest-xray-pneumonia/chest_xray/train/NORMAL")

train_data_nor["flg"] = 0



# Pneumonia

train_data_pne = pd.DataFrame({})

train_data_pne["data_id"] = os.listdir("../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA")

train_data_pne["flg"] = 1
# val_data_set

# Normal

val_data_nor = pd.DataFrame({})

val_data_nor["data_id"] = os.listdir("../input/chest-xray-pneumonia/chest_xray/val/NORMAL")

val_data_nor["flg"] = 0



# Pneumonia

val_data_pne = pd.DataFrame({})

val_data_pne["data_id"] = os.listdir("../input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA")

val_data_pne["flg"] = 1
# test_data_set

# Normal

test_data_nor = pd.DataFrame({})

test_data_nor["data_id"] = os.listdir("../input/chest-xray-pneumonia/chest_xray/test/NORMAL")

test_data_nor["flg"] = 0



# Pneumonia

test_data_pne = pd.DataFrame({})

test_data_pne["data_id"] = os.listdir("../input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA")

test_data_pne["flg"] = 1
# data shape

print("train data normal:{}".format(train_data_nor.shape), "train_data_pneumonia:{}".format(train_data_pne.shape))

print("val data normal:{}".format(val_data_nor.shape), "val_data_pneumonia:{}".format(val_data_pne.shape))

print("test data normal:{}".format(test_data_nor.shape), "test_data_pneumonia:{}".format(test_data_pne.shape))
# Combine data frame

train_data = pd.concat([train_data_nor, train_data_pne])

val_data = pd.concat([val_data_nor, val_data_pne])

test_data = pd.concat([test_data_nor, test_data_pne])



# data shape

print("train data:{}".format(train_data.shape))

print("val data:{}".format(val_data.shape))

print("test data:{}".format(test_data.shape))
# data size

size = 128



# train_data_nor

train_image_nor = []



# loading

for _id in train_data_nor["data_id"]:

    path = "../input/chest-xray-pneumonia/chest_xray/train/NORMAL/"+_id+''

    img = cv2.imread(path)

    image = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    train_image_nor.append(image)
# train_data_pne

train_image_pne = []



# loading

for _id in train_data_pne["data_id"]:

    path = "../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/"+_id+''

    img = cv2.imread(path)

    image = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    train_image_pne.append(image)
# val_data_nor

val_image_nor = []



# loading

for _id in val_data_nor["data_id"]:

    path = "../input/chest-xray-pneumonia/chest_xray/val/NORMAL/"+_id+''

    img = cv2.imread(path)

    image = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    val_image_nor.append(image)
# val_data_pne

val_image_pne = []



# loading

for _id in val_data_pne["data_id"]:

    path = "../input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA/"+_id+''

    img = cv2.imread(path)

    image = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    val_image_nor.append(image)
# test_data_nor

test_image_nor = []



# loading

for _id in test_data_nor["data_id"]:

    path = "../input/chest-xray-pneumonia/chest_xray/test/NORMAL/"+_id+''

    img = cv2.imread(path)

    image = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    test_image_nor.append(image)
# test_data_pne

test_image_pne = []



# loading

for _id in test_data_pne["data_id"]:

    path = "../input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/"+_id+''

    img = cv2.imread(path)

    image = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    test_image_pne.append(image)
fig, ax = plt.subplots(4, 4, figsize=(20,20))



for i in range(4):

    ax[0,i].imshow(train_image_nor[i])

    ax[0,i].set_title("train_image_normal")

    

    ax[1,i].imshow(train_image_pne[i])

    ax[1,i].set_title("train_image_pneumonia")

    

    ax[2,i].imshow(test_image_nor[i])

    ax[2,i].set_title("test_image_normal")

    

    ax[3,i].imshow(test_image_pne[i])

    ax[3,i].set_title("test_image_pneumonia")
train_image = train_image_nor + train_image_pne

val_image = val_image_nor + val_image_pne

test_image = test_image_nor + test_image_pne
# Training data

# data dimension

X_train = np.ndarray(shape=(len(train_image), size, size, 3), dtype=np.float32)



# change to np.ndarray

i = 0



for image in train_image:

    X_train[i] = train_image[i]

    i=i+1

    

# Scaling

X_train = X_train/255



# Checking dimension

print("Train shape:{}".format(X_train.shape))
# Val data

# data dimension

X_val = np.ndarray(shape=(len(val_image), size, size, 3), dtype=np.float32)



# change to np.ndarray

i = 0



for image in val_image:

    X_val[i] = val_image[i]

    i=i+1

    

# Scaling

X_val = X_val/255



# Checking dimension

print("val shape:{}".format(X_val.shape))
# Test data

# data dimension

X_Test = np.ndarray(shape=(len(test_image), size, size, 3), dtype=np.float32)



# change to np.ndarray

i = 0



for image in test_image:

    X_Test[i] = test_image[i]

    i=i+1

    

# Scaling

X_Test = X_Test/255



# Checking dimension

print("Test shape:{}".format(X_Test.shape))
# train target data

y_train = train_data["flg"]



# change to np.array

y_train = np.array(y_train.values)

print("y_train shape:{}".format(y_train.shape))
# val target data

y_val = val_data["flg"]



# change to np.array

y_val = np.array(y_val.values)

print("y_val shape:{}".format(y_val.shape))
# test target data

y_test = test_data["flg"]



# change to np.array

y_test = np.array(y_test.values)

print("y shape:{}".format(y_test.shape))
def define_model():

    model = Sequential()

    # 1st layer block

    model.add(BatchNormalization(input_shape=(size, size, 3)))

    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1)))

    model.add(BatchNormalization())

    model.add(Activation("relu"))

    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1)))

    model.add(BatchNormalization())

    model.add(Activation("relu"))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Dropout(0.2))

    

    # 2nd layer block

    model.add(BatchNormalization())

    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1)))

    model.add(BatchNormalization())

    model.add(Activation("relu"))

    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1)))

    model.add(BatchNormalization())

    model.add(Activation("relu"))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Dropout(0.2))

    

    # 3rd layer block

    model.add(BatchNormalization())

    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1)))

    model.add(BatchNormalization())

    model.add(Activation("relu"))

    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1)))

    model.add(BatchNormalization())

    model.add(Activation("relu"))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Dropout(0.2))

    

    # 4th layer block

    model.add(BatchNormalization())

    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1)))

    model.add(BatchNormalization())

    model.add(Activation("relu"))

    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1)))

    model.add(BatchNormalization())

    model.add(Activation("relu"))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Dropout(0.2))

    

    # Flatten

    model.add(Flatten())

    

    # Dense layer

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.3))

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.3))

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(2, activation='softmax'))

    

    # model

    model.compile(loss='sparse_categorical_crossentropy',

                  optimizer=Adam(lr=0.0001, decay=0.000001),

                  metrics=["accuracy"])

    return model
# Data augmentation

datagen = ImageDataGenerator(rotation_range=15, 

                             width_shift_range=0.2, 

                             height_shift_range=0.2,

                             horizontal_flip=True)



datagen.fit(X_train)



# Model check point

mc = ModelCheckpoint("cnn_model_01.h5",

                     monitor='val_loss',

                     save_best_only=True,

                     verbose=1)



es = EarlyStopping(monitor='val_loss',

                   patience=5)
# Calculation

model = define_model()



# Training

batch_size= 12

epochs = 100

valid_samples = 100

train_samples = len(X_train) - valid_samples



history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),

                    steps_per_epoch = train_samples / batch_size,

                    epochs = epochs,

                    callbacks = [mc, es],

                    validation_data = datagen.flow(X_val, y_val, batch_size=batch_size),

                    validation_steps = valid_samples / batch_size )
train_loss = history.history["loss"]

val_loss = history.history["val_loss"]



plt.figure(figsize=(8, 4))

plt.plot(range(len(train_loss)), train_loss, label='train_loss')

plt.plot(range(len(val_loss)), val_loss, label='valid_loss')

plt.xlabel('epoch', fontsize=16)

plt.ylabel('loss', fontsize=16)

plt.yscale("log")

plt.legend(fontsize=16)

plt.show()



train_loss = history.history["accuracy"]

val_loss = history.history["val_accuracy"]



plt.figure(figsize=(8, 4))

plt.plot(range(len(train_loss)), train_loss, label='accuracy')

plt.plot(range(len(val_loss)), val_loss, label='val_accuracy')

plt.xlabel('epoch', fontsize=16)

plt.ylabel('accuracy', fontsize=16)

plt.legend(fontsize=16)

plt.show()
# Loading best model

model = load_model('cnn_model_01.h5')



# Best model accuracy and loss

evaluation = model.evaluate(X_Test, y_test)

print('test_loss:%.3f' % evaluation[0])

print('test_accuracy:%.3f' % evaluation[1])
# predict label

y_pred = model.predict_classes(X_Test)



# Confusion matrix

cnf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)



# Visualization

fig, ax = plt.subplots(figsize=(6, 6))

ax.matshow(cnf_matrix, cmap=plt.cm.Blues, alpha=0.3)

for i in range(cnf_matrix.shape[0]): 

    for j in range(cnf_matrix.shape[1]): 

        ax.text(x=j, y=i, s=cnf_matrix[i,j], va='center', ha='center')

plt.xlabel("predicted label")

plt.ylabel("true label")

plt.show()
print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))

print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))

print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))

print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))