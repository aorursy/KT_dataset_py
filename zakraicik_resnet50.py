import numpy as np 
import pandas as pd
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
import seaborn as sns
import time
from imagehash import phash
from math import sqrt


from subprocess import check_output
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50,MobileNetV2
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(os.listdir("../input"))
def top_5_accuracy(x,y): 
    t5 = top_k_categorical_accuracy(x,y,5)
    return t5
train_imgs = "../input/train"
test_imgs = "../input/test"

resize = 224
batch_size = 64
train = pd.read_csv("../input/train.csv")
train = train.loc[train['Id'] != 'new_whale']
num_classes = len(train['Id'].unique())
d = {cat: k for k,cat in enumerate(train.Id.unique())}
plt.title('Distribution of classes excluding new_whale');
train.Id.value_counts()[1:].plot(kind='hist');
im_arrays = []
labels = []
fs = {} ##dictionary with original size of each photo 
for index, row in tqdm(train.iterrows()):  
    im = cv2.imread(os.path.join(train_imgs,row['Image']),0)
    norm_image = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    new_image = cv2.resize(norm_image,(resize,resize))
    new_image = np.reshape(new_image,[resize,resize,1])
    im_arrays.append(new_image)
    labels.append(d[row['Id']])
    fs[row['Image']] = norm_image.shape
train_ims = np.array(im_arrays)
train_labels = np.array(labels)
train_labels = keras.utils.to_categorical(train_labels)
x_train,x_val, y_train, y_val = train_test_split(train_ims,
                                                   train_labels,
                                                   test_size=0.10, 
                                                   random_state=42
                                                  )
print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)
##print(test_imgs.shape)
gen =ImageDataGenerator(zoom_range = 0.2,
                            horizontal_flip = True
                       )
reduceLROnPlat = ReduceLROnPlateau(monitor='val_top_5_accuracy',
                                      factor = 0.50,
                                      patience = 3,
                                      verbose = 1, 
                                      mode = 'max', 
                                      min_delta = .001,
                                      min_lr = 1e-5
                                  )

earlystop = EarlyStopping(monitor='val_top_5_accuracy',
                            mode= 'max',
                            patience= 5 )

callbacks = [earlystop, reduceLROnPlat]
model = ResNet50(input_shape=(resize, resize, 1),
                      weights=None, 
                      classes=num_classes)
model.compile(optimizer=Adam(lr = .005), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])
print(model.summary())
batches = gen.flow(x_train, y_train, batch_size=batch_size)
val_batches=gen.flow(x_val, y_val, batch_size=batch_size)
batches.n//batch_size
epochs = 50
history=model.fit_generator(generator=batches, 
                            steps_per_epoch=batches.n//batch_size, 
                            epochs=epochs, 
                            validation_data=val_batches, 
                            validation_steps=val_batches.n//batch_size,
                            callbacks = callbacks)
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['categorical_crossentropy'], color='b', label="Training loss")
ax[0].plot(history.history['val_categorical_crossentropy'], color='r', label="Validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['top_5_accuracy'], color='b', label="Training Top 5 Accuracy")
ax[1].plot(history.history['val_top_5_accuracy'], color='r',label="Validation Top 5 accuracy")
legend = ax[1].legend(loc='best', shadow=True)
