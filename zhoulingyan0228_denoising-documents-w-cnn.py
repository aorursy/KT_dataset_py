import numpy as np 
import scipy as sp 
import pandas as pd
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
PATCH_WIDTH_HALF = 4
PATCH_WIDTH = PATCH_WIDTH_HALF * 2 + 1

def train_patch_generator(train_imgs, train_cleaned_imgs, epochs = 5):
    for _ in range(epochs):
        for train_file, train_cleaned_file in zip(train_imgs, train_cleaned_imgs):
            patches = []
            labels = []
            train_img = cv2.imread(train_file, cv2.IMREAD_GRAYSCALE)
            train_cleaned_img = cv2.imread(train_cleaned_file, cv2.IMREAD_GRAYSCALE)
            train_cleaned_img = cv2.threshold(train_cleaned_img, 200, 255,cv2.THRESH_BINARY)[1]
            train_img_ext = cv2.copyMakeBorder(train_img, PATCH_WIDTH_HALF, PATCH_WIDTH_HALF, PATCH_WIDTH_HALF, PATCH_WIDTH_HALF, cv2.BORDER_REPLICATE)
            #thresholded_img_ext = cv2.adaptiveThreshold(train_img_ext,255,cv2.ADAPTIVE_THRESH_MEAN_C,
            #                                            cv2.THRESH_BINARY,51,30)
            #eroded_img_ext = cv2.erode(train_img_ext, np.ones((3,3),np.uint8), 1)
            #eroded_thresh_ext = cv2.erode(thresholded_img_ext, np.ones((3,3),np.uint8), 1)
            for i in range(train_img.shape[0]):
                for j in range(train_img.shape[1]):
                    label = train_cleaned_img[i][j]
                    patch_c1 = train_img_ext[i:i+PATCH_WIDTH, j:j+PATCH_WIDTH].astype(np.float32)/255.
                    #patch_c2 = thresholded_img_ext[i:i+PATCH_WIDTH, j:j+PATCH_WIDTH].astype(np.float32)/255.
                    #patch_c3 = eroded_img_ext[i:i+PATCH_WIDTH, j:j+PATCH_WIDTH].astype(np.float32)/255.
                    #patch_c4 = eroded_thresh_ext[i:i+PATCH_WIDTH, j:j+PATCH_WIDTH].astype(np.float32)/255.
                    patches.append(np.expand_dims(patch_c1, axis=2))
                    #patches.append(np.stack((patch_c1, patch_c2), axis=2))
                    labels.append(label / 255.)
            patches = np.array(patches)# patches.shape
            labels = np.array(labels) # labels.shape
            yield (patches, labels)
            
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
          activation='relu', input_shape=(PATCH_WIDTH, PATCH_WIDTH, 1)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(lr=0.002),
              metrics=['mse'])
partial_train_imgs, validate_imgs, partial_train_labels, validate_labels = train_test_split(train_imgs, train_cleaned_imgs, test_size=0.1)
# len(partial_train_imgs)
# len(validate_imgs)
EPOCHS=10
model.fit_generator(train_patch_generator(partial_train_imgs, partial_train_labels, EPOCHS), epochs=EPOCHS, steps_per_epoch=len(partial_train_labels))
score = model.evaluate_generator(train_patch_generator(validate_imgs, validate_labels, 1), steps=len(validate_labels))
print(score)
def test_patch_generator(test_imgs):
    for test_file in test_imgs:
        patches = []
        test_img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
        test_img_ext = cv2.copyMakeBorder(test_img, PATCH_WIDTH_HALF, PATCH_WIDTH_HALF, PATCH_WIDTH_HALF, PATCH_WIDTH_HALF, cv2.BORDER_REPLICATE)
        #thresholded_img_ext = cv2.adaptiveThreshold(test_img_ext,255,cv2.ADAPTIVE_THRESH_MEAN_C,
        #                                            cv2.THRESH_BINARY,51,30) 
        #eroded_img_ext = cv2.erode(train_img_ext, np.ones((3,3),np.uint8), 1)
        #eroded_thresh_ext = cv2.erode(thresholded_img_ext, np.ones((3,3),np.uint8), 1)
        for i in range(test_img.shape[0]):
            for j in range(test_img.shape[1]):
                patch_c1 = test_img_ext[i:i+PATCH_WIDTH, j:j+PATCH_WIDTH].astype(np.float32) / 255.
                #patch_c2 = thresholded_img_ext[i:i+PATCH_WIDTH, j:j+PATCH_WIDTH].astype(np.float32)/255.
                #patch_c3 = eroded_img_ext[i:i+PATCH_WIDTH, j:j+PATCH_WIDTH].astype(np.float32)/255..
                #patch_c4 = eroded_thresh_ext[i:i+PATCH_WIDTH, j:j+PATCH_WIDTH].astype(np.float32)/255..
                patches.append(np.expand_dims(patch_c1, axis=2))
                #patches.append(np.stack((patch_c1, patch_c2), axis=2))
        patches = np.array(patches)
        yield patches

def test_patch_id_generator(test_imgs):
    for test_file in test_imgs:
        id = test_file.replace('../input/test/', '').replace('.png', '')
        pids = []
        test_img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
        for i in range(test_img.shape[0]):
            for j in range(test_img.shape[1]):
                pids.append(id + '_' + str(i+1) + '_' + str(j+1)) 
        yield pids
for idx in [8]:
    img = cv2.imread(train_imgs[idx], cv2.IMREAD_GRAYSCALE)
    cleaned_img = cv2.imread(train_cleaned_imgs[idx], cv2.IMREAD_GRAYSCALE)
    predicted_mask = model.predict_generator(
        generator=test_patch_generator([train_imgs[idx]]),
        steps=1).reshape(img.shape).clip(0, 1).round()
    predicted = cv2.bitwise_and(img, 255, mask=(1-predicted_mask).astype(np.uint8))
    predicted = cv2.bitwise_or(predicted, 255, mask=predicted_mask.astype(np.uint8))
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
    plt.imshow(cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,51,30), 'gray');
    plt.title('Adaptive Threshold Cleaned')
for i, f in enumerate(test_imgs):
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    predicted = model.predict_generator(
        generator=test_patch_generator([f]),
        steps=1).clip(0, 1)
    df = pd.DataFrame({'id': [], 'value': []})
    df['id'] = next(test_patch_id_generator([f]))
    df['value'] = predicted
    if i == 0:
        df.to_csv('submission.csv', header=True, index=False)
    else:
        df.to_csv('submission.csv', header=False, mode='a', index=False)