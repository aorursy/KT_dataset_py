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
plt.figure()
plt.imshow(apply_noise(apply_jitter(faces[0])), 'gray', vmin=0, vmax=1);
N_COMPONENTS = N_CLASSES 
augments = []
for f in faces:
    augments.append(f)
    for _ in range(10):
        augments.append(apply_noise(apply_jitter(f)))
augments = np.array(augments)
pca = PCA(N_COMPONENTS)
pca.fit(augments.reshape(augments.shape[0],-1))
plt.figure(figsize=(30,20))
PLT_WH = math.ceil(math.sqrt(N_COMPONENTS))
for i in range(N_COMPONENTS):
    plt.subplot(PLT_WH, PLT_WH, i+1)
    plt.imshow(pca.components_[i].reshape((IMG_WH, IMG_WH)))
def train_generator(imgs, labels, aug_count):
    while True:
        for i in range(len(imgs)):
            pca_vecs= [pca.transform([imgs[i].flatten()])[0]]
            for _ in range(aug_count):
                pca_vecs.append(pca.transform([apply_noise(apply_jitter(imgs[i])).flatten()])[0])
            yield np.array(pca_vecs), np.array([labels[i]] *( aug_count + 1))

def steps_per_epoch(imgs, labels, aug_count):
    return len(imgs)

def test_generator(imgs):
    while True: 
        imgs_noises = []
        for i in imgs:
            imgs_noises.append(apply_noise(apply_jitter(i)))
        imgs_noises = np.array(imgs_noises)
        yield np.array(pca.transform(imgs_noises.reshape(imgs.shape[0], -1)))
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1024, activation='relu', input_dim=N_COMPONENTS))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(40, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
EPOCH = 10
AUG_COUNT = 20
model.fit_generator(train_generator(faces, labels, AUG_COUNT), epochs=EPOCH, steps_per_epoch=steps_per_epoch(faces, labels, AUG_COUNT), verbose=2)
predicted_raw = model.predict_generator(test_generator(faces), steps=1)
predicted = np.argmax(predicted_raw, axis=1)
print(accuracy_score(predicted, labels))
sns.heatmap(confusion_matrix(predicted, labels));