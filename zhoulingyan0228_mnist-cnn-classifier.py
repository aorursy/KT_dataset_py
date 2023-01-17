import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import seaborn as sns
import math
import cv2

tf.logging.set_verbosity(tf.logging.WARN)
IMG_SIZE = 28
df = pd.read_csv('../input/train.csv') #df.head()
labels = df['label'].values
features = df.drop(['label'], axis=1).values.reshape((-1, IMG_SIZE, IMG_SIZE)) / 255.
test_features = pd.read_csv('../input/test.csv').values.reshape((-1, IMG_SIZE, IMG_SIZE)) / 255.
def auto_crop(img):
    horz_margin = np.mean(img, axis=0)
    vert_margin = np.mean(img, axis=1)
    xstart = 0 
    xend = img.shape[1]
    ystart = 0 
    yend = img.shape[0]
    img2 = cv2.copyMakeBorder(img, 0, img.shape[0], 0, img.shape[1], cv2.BORDER_CONSTANT, value=[0])
    while xstart < xend-1:
        if horz_margin[xstart] > 1/img.shape[0]:
            break
        xstart += 1
    while xstart < xend-1:
        if horz_margin[xend-1] > 1/img.shape[0]:
            break
        xend -= 1
    while ystart < yend-1:
        if vert_margin[ystart] > 1/img.shape[1]:
            break
        ystart += 1
    while ystart < yend-1:
        if vert_margin[yend-1] > 1/img.shape[1]:
            break
        yend -= 1
    rect_size = max(yend - ystart, xend - xstart)

    return cv2.resize(img2[ystart:ystart+rect_size, xstart:xstart+rect_size], dsize=img.shape, interpolation=cv2.INTER_NEAREST)

def augment_dataset(feature, label, augment_count):
    aug_features = []
    aug_labels = []
    for i in range(augment_count+1):
        ch1 = feature
        if i!=0:
            JITTER = 4
            pts1 = np.array(np.random.uniform(-JITTER, JITTER, size=(4,2))+np.array([[0,0],[0,ch1.shape[1]],[ch1.shape[0],0],[ch1.shape[0],ch1.shape[1]]])).astype(np.float32)
            pts2 = np.array([[0,0],[0,ch1.shape[1]],[ch1.shape[0],0],[ch1.shape[0],ch1.shape[1]]]).astype(np.float32)

            M = cv2.getPerspectiveTransform(pts1,pts2)

            ch1 = cv2.warpPerspective(ch1,M,ch1.shape)
            ch1 = ch1 + np.random.uniform(low=-0.3, high=0.3, size=ch1.shape).clip(0, 1)
        
        aug_features.append(ch1)
        aug_labels.append(label)
    return aug_features, aug_labels
def train_generator(features, labels, augment_count=10, imgs_per_batch = 10):
    while True:
        for i in range(0, len(features), imgs_per_batch):
            features_batch = []
            labels_batch = []
            for j in range(i, min(len(features), i+imgs_per_batch)):
                cur_features, cur_labels = augment_dataset(features[j], labels[j], augment_count)
                features_batch.extend(cur_features)
                labels_batch.extend(cur_labels)
            yield np.array(features_batch),np.array(labels_batch)

def steps_per_epoch(features, imgs_per_batch=10):
    return int(math.ceil(len(features)/imgs_per_batch))

def test_generator(features):
    yield np.array(features)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Reshape((IMG_SIZE, IMG_SIZE, 1), input_shape=(IMG_SIZE, IMG_SIZE, )))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(rate=0.3))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(rate=0.3))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
EPOCHS=100
AUG_COUNT = 3
IMGS_PER_BATCH = 70
model.fit_generator(train_generator(features, labels, AUG_COUNT, IMGS_PER_BATCH),
                    epochs=EPOCHS,
                    steps_per_epoch=steps_per_epoch(features, IMGS_PER_BATCH),
                    validation_data=train_generator(features, labels, AUG_COUNT, IMGS_PER_BATCH),
                    validation_steps=steps_per_epoch(features, IMGS_PER_BATCH),
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5,monitor='val_loss')],
                    verbose=2)
X_train, X_validate, y_train, y_validate = train_test_split(features, labels, test_size=0.1)
predicted = np.argmax(model.predict_generator(test_generator(X_validate), steps=1), axis=1)
tmp = pd.DataFrame(sklearn.metrics.confusion_matrix(y_validate, predicted))
plt.subplots(figsize=(10,10)) 
sns.heatmap(tmp, annot=True, fmt='.1f')
predicted = np.argmax(model.predict_generator(test_generator(test_features), steps=1), axis=1)
out_df = pd.DataFrame({'Label': predicted})
out_df['ImageId'] = out_df.index + 1
out_df.to_csv('submission.csv', index=False)
model.save('mnist_model.h5')