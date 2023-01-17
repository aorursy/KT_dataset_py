import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow.keras import models, layers, optimizers, utils

import cv2 as cv



from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def extract_label(img_path,train = True):

    filename, _ = os.path.splitext(os.path.basename(img_path))



    subject_id, etc = filename.split('__')

    

    if train:

        gender, lr, finger, _, _ = etc.split('_')

    else:

        gender, lr, finger, _ = etc.split('_')

    

    gender = 0 if gender == 'M' else 1

    lr = 0 if lr == 'Left' else 1



    if finger == 'thumb':

        finger = 0

    elif finger == 'index':

        finger = 1

    elif finger == 'middle':

        finger = 2

    elif finger == 'ring':

        finger = 3

    elif finger == 'little':

        finger = 4

        

    return np.array([subject_id, gender, lr, finger], dtype=np.uint16)





img_size = 90



def loading_data(path,train):

    print("loading data from: ",path)

    data = []

    for img in os.listdir(path):

        try:

            img_array = cv.imread(os.path.join(path, img), cv.IMREAD_GRAYSCALE)

            img_resize = cv.resize(img_array, (img_size, img_size))

            label = extract_label(os.path.join(path, img),train)

            data.append([label[3], img_resize ])

        except Exception as e:

            pass

    data

    return data
Real_path = "../input/socofing/SOCOFing/Real"

Easy_path = "../input/socofing/SOCOFing/Altered/Altered-Easy"

Medium_path = "../input/socofing/SOCOFing/Altered/Altered-Medium"

Hard_path = "../input/socofing/SOCOFing/Altered/Altered-Hard"





Easy_data = loading_data(Easy_path, train = True)

Medium_data = loading_data(Medium_path, train = True)

Hard_data = loading_data(Hard_path, train = True)



data = np.concatenate([Easy_data, Medium_data, Hard_data], axis=0)



del Easy_data, Medium_data, Hard_data
test = loading_data(Real_path, train = False)

test = np.array(test)
print(data.shape)

print(test.shape)
X, y = [], []



for label, feature in data:

    X.append(feature)

    y.append(label)

    

del data



X = np.array(X).reshape(-1, img_size, img_size, 1)

X = X / 255.0



y = utils.to_categorical(y, num_classes = 5)
plt.figure(figsize=(15, 10))

plt.subplot(1, 4, 1)

plt.title(y[0])

plt.imshow(X[0].squeeze(), cmap='gray')

plt.subplot(1, 4, 2)

plt.title(y[1])

plt.imshow(X[1].squeeze(), cmap='gray')

plt.subplot(1, 4, 3)

plt.title(y[2])

plt.imshow(X[2].squeeze(), cmap='gray')

plt.subplot(1, 4, 4)

plt.title(y[3])

plt.imshow(X[3].squeeze(), cmap='gray')
X_test, y_test = [], []



for label, feature in test:

    X_test.append(feature)

    y_test.append(label)

    

del test    

X_test = np.array(X_test).reshape(-1, img_size, img_size, 1)

X_test = X_test / 255.0



y_test = utils.to_categorical(y_test, num_classes = 5)
plt.figure(figsize=(15, 10))

plt.subplot(1, 4, 1)

plt.title(y_test[0])

plt.imshow(X_test[0].squeeze(), cmap='gray')

plt.subplot(1, 4, 2)

plt.title(y_test[1])

plt.imshow(X_test[1].squeeze(), cmap='gray')

plt.subplot(1, 4, 3)

plt.title(y_test[2])

plt.imshow(X[2].squeeze(), cmap='gray')

plt.subplot(1, 4, 4)

plt.title(y[3])

plt.imshow(X[3].squeeze(), cmap='gray')
print(X)



print(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 1)
print("Full Data:  ",X.shape)

print("Train:      ",X_train.shape)

print("Validation: ",X_val.shape)

print("Test:       ",X_test.shape)
print("Full Data:  ",y.shape)

print("Train:      ",y_train.shape)

print("Validation: ",y_val.shape)

print("Test:       ",y_test.shape)
model = models.Sequential()



model.add(layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = 'Same', activation = 'relu', input_shape = (90, 90, 1)))

model.add(layers.MaxPooling2D(pool_size = (2, 2)))

model.add(layers.Dropout(0.25))



model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))

model.add(layers.MaxPooling2D(pool_size = (2, 2)))

model.add(layers.Dropout(0.25))



model.add(layers.Flatten())

model.add(layers.Dense(100, activation = 'relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(5, activation = 'softmax'))
model.summary()
model.compile(

    optimizer = 'adam',

    loss = 'categorical_crossentropy',

    metrics = ['accuracy']

)
history = model.fit(

    X_train,

    y_train,

    epochs = 30,

    batch_size = 128,

    validation_data = (X_val, y_val)

)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, label='Training acc')

plt.plot(epochs, val_acc, label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss,  label='Training loss')

plt.plot(epochs, val_loss, label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



score = model.evaluate([X_test], [y_test], verbose=0)

print("Score: ",score[1]*100)



plt.show()
pred = model.predict([X_test])
print(pred[0])

print(y_test[0])