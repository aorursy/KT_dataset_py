import numpy as np 
import scipy as sp 
import pandas as pd
import math
import matplotlib.pyplot as plt
import glob 
import cv2
import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import tensorflow as tf

train_imgs = glob.glob("../input/train/*.png")
train_imgs.sort()
train_cleaned_imgs = glob.glob("../input/train_cleaned/*.png")
train_cleaned_imgs.sort()
test_imgs= glob.glob("../input/test/*.png")
def total_batches(files):
    total = 0
    for f in files:
        total += cv2.imread(f, cv2.IMREAD_GRAYSCALE).shape[0]
    return total
total_ttrain_pixels = total_batches(train_imgs)
total_test_pixels = total_batches(test_imgs)
PATCH_WIDTH_HALF = 4
PATCH_WIDTH = PATCH_WIDTH_HALF * 2 + 1

def train_patch_generator(train_imgs, train_cleaned_imgs, augments=5, epochs = 5):
    for _ in range(epochs):
        for train_file, train_cleaned_file in zip(train_imgs, train_cleaned_imgs):
            train_img = cv2.imread(train_file, cv2.IMREAD_GRAYSCALE)
            train_cleaned_img = cv2.imread(train_cleaned_file, cv2.IMREAD_GRAYSCALE)
            train_cleaned_img = cv2.threshold(train_cleaned_img, 200, 255,cv2.THRESH_BINARY)[1]
            train_img_ext = cv2.copyMakeBorder(train_img, PATCH_WIDTH_HALF, PATCH_WIDTH_HALF, PATCH_WIDTH_HALF, PATCH_WIDTH_HALF, cv2.BORDER_REPLICATE)
            mean_vert_ext = np.mean(train_img_ext, axis=1)
            #thresholded_img_ext = cv2.adaptiveThreshold(train_img_ext,255,cv2.ADAPTIVE_THRESH_MEAN_C,
            #                             cv2.THRESH_BINARY,31,30) 
            morph_grad_ext = cv2.morphologyEx(train_img_ext, cv2.MORPH_GRADIENT, np.ones((5,5),np.uint8))
            for i in range(train_img.shape[0]):
                patches = []
                labels = []
                weights = []
                vert_margins = []
                features = []
                for j in range(train_img.shape[1]):
                    label = train_cleaned_img[i][j]
                    patch_c1 = train_img_ext[i:i+PATCH_WIDTH, j:j+PATCH_WIDTH].astype(np.float32)/255.
                    vert_margin = mean_vert_ext[i:i+PATCH_WIDTH].astype(np.float32)/255.
                    #patches.append(np.expand_dims(patch_c1, axis=2))
                    weight = morph_grad_ext[i+PATCH_WIDTH_HALF, j+PATCH_WIDTH_HALF]/255.*0.5+0.5
                    vert_margins.append(vert_margin)
                    #patches.append(np.stack((patch_c1, patch_c2), axis=2))
                    patches.append(np.expand_dims(patch_c1, axis=2))
                    labels.append(label / 255.)
                    weights.append(weight)
                    for _ in range(augments):
                        #patches.append(np.expand_dims(patch_c1, axis=2))
                        vert_margins.append(vert_margin)
                        #patches.append(np.stack((patch_c1, patch_c2), axis=2))
                        patches.append(np.expand_dims(patch_c1+np.random.normal(scale=0.2, size=patch_c1.shape), axis=2))
                        labels.append(label / 255.)
                        weights.append(weight)
                patches = np.array(patches)# patches.shape
                vert_margins = np.array(vert_margins) # 
                labels = np.array(labels) # labels.shape
                weights = np.array(weights) #
                yield ([patches , vert_margins], labels, weights)
            
x1 = tf.keras.layers.Input(name='patch', shape=(PATCH_WIDTH, PATCH_WIDTH, 1))
dr1 = tf.keras.layers.Dropout(rate=0.1)(x1)
cv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(dr1)
mp1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(cv1)
f1 = tf.keras.layers.Flatten()(mp1)
x2 = tf.keras.layers.Input(name='vert_margin', shape=(PATCH_WIDTH,))
cc = tf.keras.layers.concatenate([f1, x2])
d1 = tf.keras.layers.Dense(1024, activation='relu')(cc)
d2 = tf.keras.layers.Dense(512, activation='relu')(d1)
d3 = tf.keras.layers.Dense(256, activation='relu')(d2)
dr2 = tf.keras.layers.Dropout(rate=0.5)(d3)
out = tf.keras.layers.Dense(1, activation='sigmoid')(dr2)

model = tf.keras.models.Model(inputs=[x1,x2], outputs=out)
model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['mse'])
EPOCHS=2
AUGMENTS = 2
model.fit_generator(train_patch_generator(train_imgs, train_cleaned_imgs, AUGMENTS, EPOCHS), epochs=EPOCHS, steps_per_epoch=total_batches(train_imgs), verbose=2)
def test_patch_generator(test_imgs):
    for test_file in test_imgs:
        test_img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
        test_img_ext = cv2.copyMakeBorder(test_img, PATCH_WIDTH_HALF, PATCH_WIDTH_HALF, PATCH_WIDTH_HALF, PATCH_WIDTH_HALF, cv2.BORDER_REPLICATE)
        mean_vert_ext = np.mean(test_img_ext, axis=1)
        #thresholded_img_ext = cv2.adaptiveThreshold(test_img_ext,255,cv2.ADAPTIVE_THRESH_MEAN_C,
        #                                            cv2.THRESH_BINARY,31,30) 
        #eroded_img_ext = cv2.erode(train_img_ext, np.ones((3,3),np.uint8), 1)
        #eroded_thresh_ext = cv2.erode(thresholded_img_ext, np.ones((3,3),np.uint8), 1)
        for i in range(test_img.shape[0]):
            patches = []
            vert_margins = []
            for j in range(test_img.shape[1]):
                patch_c1 = test_img_ext[i:i+PATCH_WIDTH, j:j+PATCH_WIDTH].astype(np.float32) / 255.
                vert_margin = mean_vert_ext[i:i+PATCH_WIDTH].astype(np.float32)/255.
                #patch_c3 = eroded_img_ext[i:i+PATCH_WIDTH, j:j+PATCH_WIDTH].astype(np.float32)/255..
                #patch_c4 = eroded_thresh_ext[i:i+PATCH_WIDTH, j:j+PATCH_WIDTH].astype(np.float32)/255..
                patches.append(np.expand_dims(patch_c1, axis=2))
                #patches.append(np.stack((patch_c1, patch_c2), axis=2))
                vert_margins.append(vert_margin)
            patches = np.array(patches)
            vert_margins = np.array(vert_margins)
            yield [patches, vert_margins]

def test_patch_id_generator(test_imgs):
    pids = []
    for test_file in test_imgs:
        id = test_file.replace('../input/test/', '').replace('.png', '')
        test_img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
        for i in range(test_img.shape[0]):
            for j in range(test_img.shape[1]):
                pids.append(id + '_' + str(i+1) + '_' + str(j+1)) 
    yield pids
for idx in [8]:
    img = cv2.imread(train_imgs[idx], cv2.IMREAD_GRAYSCALE)
    cleaned_img = cv2.imread(train_cleaned_imgs[idx], cv2.IMREAD_GRAYSCALE)
    th_mask = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                                    cv2.THRESH_BINARY,31,30) 
    predicted_mask = model.predict_generator(
        generator=test_patch_generator([train_imgs[idx]]),
        steps=total_batches([train_cleaned_imgs[idx]])).reshape(img.shape).clip(0, 1).round().astype(np.uint8)
    background = np.full(img.shape, 255, dtype=np.uint8)
    predicted = cv2.bitwise_or(img, 0, dst=background, mask=(1-predicted_mask))
    plt.figure(figsize=(60,30))
    plt.subplot(2,2,1)
    plt.imshow(img, 'gray');
    plt.title('Uncleaned')
    plt.subplot(2,2,2)
    plt.imshow(cleaned_img, 'gray');
    plt.title('Manually Cleaned')
    plt.subplot(2,2,3)
    plt.imshow(predicted, 'gray');
    plt.title('Auto Cleaned')
    plt.subplot(2,2,4)
    plt.imshow(cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,31,30), 'gray');
    plt.title('Adaptive Threshold Cleaned')
for i, f in enumerate(test_imgs):
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE) 
    predicted_mask = model.predict_generator(
        generator=test_patch_generator([f]),
        steps=img.shape[0]).reshape(img.shape).clip(0, 1).round().astype(np.uint8)
    background = np.full(img.shape, 255, dtype=np.uint8)
    predicted = cv2.bitwise_or(img, 0, dst=background, mask=(1-predicted_mask))
    predicted = predicted/255.
    df = pd.DataFrame({'id': [], 'value': []})
    df['id'] = next(test_patch_id_generator([f]))
    df['value'] = predicted.flatten()
    if i == 0:
        df.to_csv('submission.csv', header=True, index=False)
    else:
        df.to_csv('submission.csv', header=False, mode='a', index=False)