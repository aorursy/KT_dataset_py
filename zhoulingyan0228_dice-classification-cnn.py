import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import random
import cv2
import tensorflow as tf
import seaborn as sns
import math
def read_vectors(*filenames):
    data = np.vstack(
        tuple(np.fromfile(filename, dtype=np.uint8).reshape(-1,401)
                      for filename in filenames))
    return data[:,1:], data[:,0]

X_train, y_train = read_vectors(*[
    "../input/snake-eyes/snakeeyes_{:02d}.dat".format(nn) for nn in range(2)])
X_train = (X_train.reshape((X_train.shape[0], 20, 20))/255.).astype(np.float32)
X_test, y_test = read_vectors("../input/snake-eyes/snakeeyes_test.dat")
X_test = (X_test.reshape((X_test.shape[0], 20, 20))/255.).astype(np.float32)
IMG_HW = 20
class CountEncoder:
    def fit(self,y):
        pass
    def transform(self,y):
        return y-1
    def fit_transform(self,y):
        return transform(y)
    def inverse_transform(self,y):
        return y+1
labelEncoder = CountEncoder()
y_train = labelEncoder.transform(y_train)
y_test = labelEncoder.transform(y_test)
def process_img(img):
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel=np.ones((3,3)),iterations=1,borderValue=0)
def rotate_img(img, angle=90):
    M = cv2.getRotationMatrix2D((img.shape[0]/2,img.shape[1]/2),90,1)
    return cv2.warpAffine(img,M,img.shape)
def jitter_img(img, max_jitter=3):
    M = np.eye(2,3)
    M[0][2] = random.randint(-max_jitter,max_jitter)
    M[1][2] = random.randint(-max_jitter,max_jitter)
    return cv2.warpAffine(img,M,img.shape)
def train_generator(X, y, batch_size):
    i = 0
    maxIdx = len(X)
    while True:
        X_batch = []
        y_batch = []
        for _ in range(batch_size):   
            proc = process_img(X[i])
            X_batch.append(proc)
            y_batch.append(y[i])
            X_batch.append(rotate_img(proc, 90))
            y_batch.append(y[i])
            X_batch.append(rotate_img(proc, 180))
            y_batch.append(y[i])
            X_batch.append(rotate_img(proc, 270))
            y_batch.append(y[i])
            X_batch.append(cv2.flip(proc,0))
            y_batch.append(y[i])
            X_batch.append(cv2.flip(proc,1))
            y_batch.append(y[i])
            X_batch.append(cv2.flip(proc,-1))
            y_batch.append(y[i])
            curlen = len(y_batch)
            for j in range(curlen):
                X_batch.append(jitter_img(X_batch[j]))
                y_batch.append(y_batch[j])
            i = (i+1) % maxIdx
        yield np.array(X_batch), np.array(y_batch)
        
def steps_per_epoch(X, batch_size):
    return math.ceil(len(X)/batch_size)

def test_generator(X):
    while True:
        for x in X:
            yield np.array([process_img(x)])
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Reshape((IMG_HW, IMG_HW, 1), input_shape=(IMG_HW, IMG_HW, )))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(5,5), strides=(2,2), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(5,5), strides=(2,2), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2048, activation='relu'))
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(12, activation='softmax'))
          
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), 
              metrics=['accuracy'], 
              loss='sparse_categorical_crossentropy')
EPOCHS=3
BATCH_SIZE = 5
model.fit_generator(train_generator(X_train, y_train, BATCH_SIZE), epochs=EPOCHS, steps_per_epoch=steps_per_epoch(X_train, BATCH_SIZE), verbose=2)
predicted_probs = model.predict_generator(test_generator(X_test), steps=len(X_test))
predicted_classes = np.argmax(predicted_probs, axis=1)

print(accuracy_score(y_test, predicted_classes))
sns.heatmap(confusion_matrix(y_test, predicted_classes))