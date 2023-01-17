import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sklearn
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import cv2
import sklearn
import tensorflow as tf
import seaborn as sns
import random
faces_raw = np.load('../input/olivetti_faces.npy')
labels_raw = np.load('../input/olivetti_faces_target.npy')
N_CLASSES=len(np.unique(labels_raw))
IMG_WH = 64

shuffleIdx = np.arange(len(faces_raw))
np.random.shuffle(shuffleIdx)
faces = faces_raw[shuffleIdx]
labels = labels_raw[shuffleIdx]

plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(faces[0], 'gray', vmin=0, vmax=1);
plt.subplot(1,2,2)
pd.Series(labels).plot.hist(bins=40, rwidth=0.8);
JITTER = 5

def apply_jitter(img):
    pts1 = np.array(np.random.uniform(-JITTER, JITTER, size=(4,2))+np.array([[0,0],[0,IMG_WH],[IMG_WH,0],[IMG_WH,IMG_WH]])).astype(np.float32)
    pts2 = np.array([[0,0],[0,IMG_WH],[IMG_WH,0],[IMG_WH,IMG_WH]]).astype(np.float32)
    M = cv2.getPerspectiveTransform(pts1,pts2)
    return cv2.warpPerspective(img,M,(IMG_WH,IMG_WH))
def apply_noise(img):
    #img = img + np.random.uniform(low=-0.05, high=0.05, size=img.shape)
    img = img * np.random.uniform(low=0.93, high=1.07, size=img.shape)
    img = img.clip(0, 1)
    return img
def apply_occlusion(img):
    mask = np.ones(img.shape)
    cv2.rectangle(mask, (0, 0), (img.shape[0]//random.randint(3,10), img.shape[1]), 0, thickness=-1)
    M = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),random.randint(0,360),1)
    mask = cv2.warpAffine(mask,M,mask.shape, borderValue=1)
    return cv2.bitwise_or(img, img, mask=mask.astype(np.uint8))

plt.figure()
plt.imshow(apply_occlusion(apply_noise(apply_jitter(faces[0]))), 'gray', vmin=0, vmax=1);
def train_generator(imgs, labels, aug_count):
    while True:
        for i in range(len(imgs)):
            augmts = [imgs[i]]
            for _ in range(aug_count):
                augmts.append(apply_occlusion(apply_noise(apply_jitter(imgs[i]))))
            yield np.array(augmts), np.array([labels[i]] *( aug_count + 1))

def steps_per_epoch(imgs, labels, aug_count):
    return len(imgs)

def test_generator(imgs):
    while True: 
        augmts = []
        for i in imgs:
            augmts.append(apply_occlusion(apply_noise(apply_jitter(i))))
        augmts = np.array(augmts)
        yield np.array(augmts)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Reshape((IMG_WH, IMG_WH, 1), input_shape=(IMG_WH, IMG_WH,)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(512))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(40, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
EPOCH = 30
AUG_COUNT = 50
model.fit_generator(train_generator(faces, labels, AUG_COUNT), epochs=EPOCH, steps_per_epoch=steps_per_epoch(faces, labels, AUG_COUNT), verbose=2)
predicted_raw = model.predict_generator(test_generator(faces), steps=1)
predicted = np.argmax(predicted_raw, axis=1)
print(accuracy_score(predicted, labels))
sns.heatmap(confusion_matrix(predicted, labels));