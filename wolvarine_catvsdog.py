# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2
import matplotlib.pyplot as plt
training=[]
cats=os.listdir('../input/cat-and-dog/training_set/training_set/cats')
path='../input/cat-and-dog/training_set/training_set/cats'
dogs=os.listdir('../input/cat-and-dog/training_set/training_set/dogs')
path2='../input/cat-and-dog/training_set/training_set/dogs'


categories = []
try:
    for cat in cats:
        if cat.split('.')[0]=='cat':
            arr=cv2.imread(os.path.join(path,cat))
            arr=cv2.resize(arr,(64,64))
#             arr=np.reshape(arr,(64,64,3))
            training.append([arr,1])

    for dog in dogs:
        if dog.split('.')[0]=='dog':
            arr=cv2.imread(os.path.join(path2,dog))
            arr=cv2.resize(arr,(64,64))
#             arr=np.reshape(arr,(64,64,3))
            training.append([arr,0])
except Exception as e:
    print(e)

# print((training.shape))

import random
random.shuffle(training)
X=[]
Y=[]
for feature,label in training:
    X.append(feature)
    Y.append(label)
X=np.array(X).reshape(-1,64,64,3)

    
# plt.imshow(X[1])
# X=X[:,:,:,0]
X=X/255
print(X.shape)
plt.imshow(X[2])
Y=np.array(Y).reshape(-1,1)
print(Y)

test=[]
path='../input/cat-and-dog/test_set/test_set/cats'
for cat in os.listdir(path):
    if cat.split('.')[0]=='cat':
        arr=cv2.imread(os.path.join(path,cat))
        arr=cv2.resize(arr,(64,64))
        test.append([arr,1])
path='../input/cat-and-dog/test_set/test_set/dogs'
for dog in os.listdir(path):
    if cat.split('.')[0]=='dog':
        arr=cv2.imread(os.path.join(path,dog))
        arr=cv2.resize(arr,(64,64))
        test.append([arr,0])
import random
random.shuffle(training)
X_test=[]
Y_test=[]
for feature,label in training:
    X_test.append(feature)
    Y_test.append(label)
X_test=np.array(X_test).reshape(-1,64,64,3)
Y_test=np.array(Y_test).reshape(-1,1)
print(Y_test.shape)
import tensorflow as tf
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32,(3,3),activation='relu', input_shape=(64,64,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
    ])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

history=model.fit(X,Y,epochs=10)


p=model.evaluate(X_test,Y_test)
print(Y_train.shape)
index=9

img=cv2.imread('')
plt.imshow(img)

img=cv2.resize(img,(64,64))

img=np.array(img).reshape(64,64,3)

img = np.expand_dims(img, axis=0)
p=model.predict(img)

print(np.argmax(p))
# 1-Cat
# 0-Dog
