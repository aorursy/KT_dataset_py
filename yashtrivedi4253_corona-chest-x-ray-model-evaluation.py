

import random
import os
import glob
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

from tqdm import tqdm

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,Flatten,BatchNormalization,AvgPool2D
from keras.models import Sequential,save_model,load_model
from keras.optimizers import Adam

from keras.utils.np_utils import to_categorical

from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

from sklearn.metrics import confusion_matrix,classification_report

path = "/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/"
train_dir = os.path.join(path,"train")
test_dir = os.path.join(path,"test")
print("Total num of train Images are: ",str(len(os.listdir(train_dir))))
print()
print("Total num of test Images are: ",str(len(os.listdir(test_dir))))
BATCH_SIZE = 50
IMG_SHAPE = 256
def get_label(img):
    name = img.split("-")[0]
    if name == "NORMAL2": return 0
    elif name == "IM" : return 0
    else: return 1
    
def creat_train_data():
    train = []
    #normal_train_img = []
    #corona_test_img = []
    for img in tqdm(os.listdir(train_dir)):
        label = get_label(img)
        img_path = os.path.join(train_dir,img)
        image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image,(256,256))
        image = np.array(image).reshape(-1,IMG_SHAPE,IMG_SHAPE,1)
        train.append([np.array(image),np.array(label)])
    random.shuffle(train)
    #np.save("train_data.npy",train)
    
    return train
train_dataset = creat_train_data()
len(train_dataset)
def creat_test_data():
    test = []
    for img in tqdm(os.listdir(test_dir)):
        label = get_label(img)
        img_path = os.path.join(test_dir,img)
        image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image,(256,256))
        image = np.array(image).reshape(-1,IMG_SHAPE,IMG_SHAPE,1)
        test.append([np.array(image),np.array(label)])
    random.shuffle(test)
    np.save("test_data.npy",test)
    
    return test
test_dataset = creat_test_data()

len(test_dataset)
X_test = np.array([i[0] for i in test_dataset]).reshape(-1,IMG_SHAPE,IMG_SHAPE,1)
Y_test = np.array([i[1] for i in test_dataset])
print(X_test.shape)
print(Y_test.shape)
X_train = np.array([i[0] for i in train_dataset]).reshape(-1,IMG_SHAPE,IMG_SHAPE,1)
Y_train = np.array([i[1] for i in train_dataset])
print(X_train.shape)
print(Y_train.shape)
model = Sequential()

model.add(Conv2D(filters=32,kernel_size=5,activation="relu",input_shape=(IMG_SHAPE,IMG_SHAPE,1)))
model.add(Conv2D(filters=32,kernel_size=5,activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(filters=64,kernel_size=5,activation="relu"))
model.add(Conv2D(filters=64,kernel_size=5,activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Dropout(0.2))

model.add(Conv2D(filters=128,kernel_size=5,activation="relu"))
model.add(Conv2D(filters=128,kernel_size=5,activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(filters=256,kernel_size=5,activation="relu"))
model.add(Conv2D(filters=256,kernel_size=5,activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(filters=1024,kernel_size=5,activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.25))

model.add(Dense(units=1024,activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(units=128,activation="relu"))
model.add(Dense(units=2, activation="softmax"))
model.compile(Adam(lr=0.001),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.summary()
checkpoint = ModelCheckpoint(filepath = "/",
    monitor='val_loss',
    verbose=1,
    mode = "min",
    save_best_only=True)

earlystop = EarlyStopping(monitor='val_loss',
    min_delta=0,
    patience=9,
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
    factor=0.2,
    patience=6,
    verbose=1,
    min_delta=0.0001)

#callbacks = [checkpoint, earlystop, reduce_lr]
callbacks = [earlystop, reduce_lr]
h = model.fit(x=X_train,
    y=Y_train,
    batch_size=BATCH_SIZE,
    epochs=65,
    verbose=1,
    validation_split=0.15,
    callbacks=callbacks,
    shuffle=True)
plt.plot(h.history["accuracy"])
plt.plot(h.history["val_accuracy"])
plt.legend(["accuracy","val_accuracy"])
plt.xlabel("Epochs")
plt.title("Accuracy")
plt.show()
plt.plot(h.history["loss"])
plt.plot(h.history["val_loss"])
plt.legend(["loss","val_loss"])
plt.xlabel("Epochs")
plt.title("Loss")
plt.show()
evaluation = model.evaluate(X_test,Y_test,verbose=1)
print("Model Test Loss is:",evaluation[0])
print()
print("Model Test accuracy is: "+ str(round(evaluation[1],4)*100)+ "%")
predictions = model.predict(X_test)
print(predictions)
Y_test = to_categorical(Y_test,2)
Y_test
print(confusion_matrix(Y_test.argmax(axis=1),predictions.argmax(axis=1)))


