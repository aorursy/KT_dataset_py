import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tqdm import tqdm

import seaborn as sbn
import matplotlib.image as img
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import os
import gc
import math
LABELS = {
    "Swiping Right": 0,
    "Swiping Left": 1,
    "No gesture": 2,
    "Thumb Up": 3,
}
BASE_PATH = '../input/20bn-jester'
TRAIN_DATA_CSV = BASE_PATH + '/Train.csv'
TEST_DATA_CSV = BASE_PATH + '/Test.csv'
VAL_DATA_CSV = BASE_PATH + '/Validation.csv'

TRAIN_SAMPLES_PATH = BASE_PATH + '/Train/'
TEST_SAMPLES_PATH = BASE_PATH + '/Test/'
VAL_SAMPLES_PATH = BASE_PATH + '/Validation/'
targets = pd.read_csv(TRAIN_DATA_CSV)
targets = targets[targets['label'].isin(LABELS.keys())]
targets['label'] = targets['label'].map(LABELS)
targets = targets[['video_id', 'label']]
targets = targets.reset_index()
targets
targets_validation = pd.read_csv(VAL_DATA_CSV)
targets_validation = targets_validation[targets_validation['label'].isin(LABELS.keys())]
targets_validation['label'] = targets_validation['label'].map(LABELS)
targets_validation = targets_validation[['video_id', 'label']]
targets_validation = targets_validation.reset_index()
targets_validation
def rgb2gray(rgb):
    """
    Converts numpy array of RGB to grayscale
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
def resize_frame(frame):
    """
    Resizes frames to (64, 64)
    """
    frame = img.imread(frame)
    frame = cv2.resize(frame, (64, 64))
    return frame
hm_frames = 30 # number of frames
def get_unify_frames(path):
    """
    Unifies number of frames for each training
    """
    offset = 0
    # pick frames
    frames = os.listdir(path)
    frames_count = len(frames)
    # unify number of frames 
    if hm_frames > frames_count:
        # duplicate last frame if video is shorter than necessary
        frames += [frames[-1]] * (hm_frames - frames_count)
    elif hm_frames < frames_count:
        # If there are more frames, then sample starting offset
        # diff = (frames_count - hm_frames)
        # offset = diff-1 
        frames = frames[0:hm_frames]
    return frames
# Adjust training data
train_targets = [] # training targets 
test_targets = [] # testing targets

new_frames = [] # training data after resize & unify
new_frames_test = [] # testing data after resize & unify

for idx, row in tqdm(targets.iterrows(), total=len(targets)):
    if idx % 4 == 0:
        continue
    
    partition = [] # one training
    # Frames in each folder
    frames = get_unify_frames(TRAIN_SAMPLES_PATH + str(row['video_id']))
    if len(frames) == hm_frames: # just to be sure
        for frame in frames:
            frame = resize_frame(TRAIN_SAMPLES_PATH + str(row['video_id']) + '/' + frame)
            partition.append(rgb2gray(frame))
            if len(partition) == 15: # partition each training on two trainings.
                if idx % 6 == 0:
                    new_frames_test.append(partition) # append each partition to training data
                    test_targets.append(row['label'])
                else:
                    new_frames.append(partition) # append each partition to test data
                    train_targets.append(row['label'])
                partition = []

train_data = np.asarray(new_frames, dtype=np.float16)
del new_frames[:]
del new_frames

test_data = np.asarray(new_frames_test, dtype=np.float16)
del new_frames_test[:]
del new_frames_test

gc.collect()
# we do the same for the validation data
cv_targets = []
new_frames_cv = []
for idx, row in tqdm(targets_validation.iterrows(), total=len(targets_validation)):
    if idx % 4 == 0:
        continue

    partition = []
    # Frames in each folder
    frames = get_unify_frames(VAL_SAMPLES_PATH+str(row["video_id"]))
    for frame in frames:
        frame = resize_frame(VAL_SAMPLES_PATH+str(row["video_id"])+'/'+frame)
        partition.append(rgb2gray(frame))
        if len(partition) == 15:
            new_frames_cv.append(partition)
            cv_targets.append(row['label'])
            partition = []
                
cv_data = np.array(new_frames_cv, dtype=np.float16)
del new_frames_cv[:]
del new_frames_cv
gc.collect()
print(f"Training = {len(train_data)}/{len(train_targets)} samples/labels")
print(f"Test = {len(test_data)}/{len(test_targets)} samples/labels")
print(f"Validation = {len(cv_data)}/{len(cv_targets)} samples/labels")
# Normalisation: training
print('old mean', train_data.mean())

scaler = StandardScaler(copy=False)
scaled_images  = scaler.fit_transform(train_data.reshape(-1, 15*64*64))
del train_data
print('new mean', scaled_images.mean())

scaled_images  = scaled_images.reshape(-1, 15, 64, 64, 1)
print(scaled_images.shape)
# Normalisation: test
print('old mean', test_data.mean())

scaler = StandardScaler(copy=False)
scaled_images_test = scaler.fit_transform(test_data.reshape(-1, 15*64*64))
del test_data
print('new mean', scaled_images_test.mean())

scaled_images_test = scaled_images_test.reshape(-1, 15, 64, 64, 1)
print(scaled_images_test.shape)
# Normalisation: validation
print('old mean', cv_data.mean())

scaler = StandardScaler(copy=False)
scaled_images_cv  = scaler.fit_transform(cv_data.reshape(-1, 15*64*64))
del cv_data
print('new mean',scaled_images_cv.mean())

scaled_images_cv  = scaled_images_cv.reshape(-1, 15, 64, 64, 1)
print(scaled_images_cv.shape)
del scaler
y_train = np.array(train_targets, dtype=np.int8)
y_test = np.array(test_targets, dtype=np.int8)
y_val = np.array(cv_targets, dtype=np.int8)
del train_targets
del test_targets
del cv_targets
x_train = scaled_images
x_test = scaled_images_test
x_val = scaled_images_cv
del scaled_images
del scaled_images_test
del scaled_images_cv
gc.collect()
class Conv3DModel(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
    
        # Convolutions
        self.conv1 = tf.compat.v2.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', name="conv1", data_format='channels_last')
        self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last')
        self.conv2 = tf.compat.v2.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', name="conv2", data_format='channels_last')
        self.pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2,2), data_format='channels_last')
        self.conv3 = tf.compat.v2.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', name="conv3", data_format='channels_last')
        self.pool3 = tf.keras.layers.MaxPool3D(pool_size=(2, 2,2), data_format='channels_last')
   
        # LSTM & Flatten
        self.convLSTM =tf.keras.layers.ConvLSTM2D(40, (3, 3))
        self.flatten =  tf.keras.layers.Flatten(name="flatten")

        # Dense layers
        self.d1 = tf.keras.layers.Dense(128, activation='relu', name="d1")
        self.out = tf.keras.layers.Dense(4, activation='softmax', name="output")

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.convLSTM(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.out(x)
model = Conv3DModel()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics = ['accuracy'])
history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    batch_size=32,
                    epochs=5)
model.save_weights('weights/w.tf', save_format='tf')
x_test.shape
y_test.shape
np.unique(y_test, return_counts=True)
y_pred = model.predict(x_test)
y_pred.shape
y_pred = np.argmax(y_pred, axis=-1)
y_pred.shape
y_pred
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
labels = list(LABELS.keys())
labels
cm = confusion_matrix(y_test, y_pred, normalize='true')
df_cm = pd.DataFrame(cm, range(4), range(4))
plt.figure(figsize=(10,7))
sbn.set(font_scale=1.4) # for label size
sbn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, xticklabels=labels, yticklabels=labels)

plt.show()
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred, average='macro')
recall_score(y_test, y_pred, average='macro')
f1_score(y_test, y_pred, average='macro')