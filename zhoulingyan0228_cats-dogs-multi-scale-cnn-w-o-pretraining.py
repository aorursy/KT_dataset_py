import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import IPython.display as ipd
import keras
from keras.models import *
from keras.callbacks import *
from keras.layers import *
from keras.preprocessing.image import random_brightness,random_rotation,random_shear,random_shift,random_zoom
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import skimage
import skimage.transform
import skimage.color
import imageio
import matplotlib.pyplot as plt
import seaborn as sns
import os

TRAIN_DIR = "../input/train/"
TEST_DIR = "../input/test/"

train = pd.DataFrame()
train['file'] = os.listdir(TRAIN_DIR)
train['class'] = train['file'].apply(lambda x: x.split('.')[0])
train['class_id'] = train['class'].apply(lambda x: 0 if x=='cat' else 1)
test = pd.DataFrame()
test['file'] = os.listdir(TEST_DIR)
test['id'] = test['file'].apply(lambda x: x.split('.')[0])
test['label'] = 0.5

train.head()
sns.countplot(x='class', data=train);
def image_generator(files, base_dir, labels=None, size=(256,256), batch_size=32, random_preproc=False, rotation_range=0, shear_range=0, shift_range=(0,0), zoom_range=(1,1)):
    while True:
        for i in range(0, len(files), batch_size):
            img_batch = []
            label_batch = []
            for j in range(i, min(len(files), i+batch_size)):
                img = imageio.imread(base_dir+files[j])
                img = skimage.transform.resize(img, size, mode='reflect')
                img = skimage.color.rgb2gray(img).reshape(size[0], size[1], 1)
                if random_preproc:
                    img = random_zoom(img, zoom_range,row_axis=0,col_axis=1,channel_axis=2)
                    img = random_shear(img, shear_range,row_axis=0,col_axis=1,channel_axis=2)
                    img = random_shift(img, shift_range[0],shift_range[1],row_axis=0,col_axis=1,channel_axis=2)
                    img = random_rotation(img, rotation_range,row_axis=0,col_axis=1,channel_axis=2)
                img_batch.append(img)
                if labels is not None:
                    label_batch.append(labels[j])
            
            if labels is not None:
                yield np.array(img_batch), np.array(label_batch)
            else:
                yield np.array(img_batch)
        
def make_model(size=(256,256)):
    def make_cnn(kernel_nums, x):
        for n in kernel_nums:
            x = Conv2D(n, kernel_size=3, strides=1, activation='relu', padding='same')(x)
            x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = SpatialDropout2D(0.3)(x)
        return Flatten()(x)
    inp = Input((size[0],size[1],1))
    kernel_nums = [64, 64,128,128,256,256,512,512]
    scaled = inp
    cnn_outs = []
    for i in range(6):
        scaled = AveragePooling2D(pool_size=2**i, strides=2**i)(inp)
        cnn_outs.append(make_cnn(kernel_nums[:len(kernel_nums)-i], scaled))
    x = concatenate(cnn_outs)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inp, out)
SIZE = (256,256)
model = make_model(size=SIZE)
model.summary()
#keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
#ipd.Image(filename='model.png')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
batch_size=150
X_train, X_valid, y_train, y_valid = train_test_split(train['file'].values, train['class_id'].values, test_size=0.1)
model.fit_generator(image_generator(X_train, TRAIN_DIR, labels=y_train, size=SIZE, batch_size=batch_size, random_preproc=False, rotation_range=10, shear_range=5, shift_range=(0.1,0.1), zoom_range=(0.8,1.2)),
                    epochs=25,
                    steps_per_epoch=int(math.ceil(len(y_train)/batch_size)),
                    validation_data=image_generator(X_valid, TRAIN_DIR, labels=y_valid, size=SIZE, batch_size=batch_size, random_preproc=False),
                    validation_steps=int(math.ceil(len(y_valid)/batch_size)),
                    callbacks=[EarlyStopping(monitor='val_loss',patience=3,verbose=0)],
                    verbose=1
                   )
predicted_probs = model.predict_generator(image_generator(X_valid, TRAIN_DIR, size=SIZE, batch_size=batch_size, random_preproc=False),
                                             steps=int(math.ceil(len(y_valid)/batch_size))
                                            )
predicted = np.round(predicted_probs)
print(classification_report(y_valid, predicted))
print(log_loss(y_valid, predicted_probs))
sns.heatmap(confusion_matrix(y_valid, predicted), annot=True);
predicted_probs = model.predict_generator(image_generator(test['file'], TEST_DIR, size=SIZE, batch_size=batch_size, random_preproc=False),
                                             steps=int(math.ceil(len(test['file'])/batch_size))
                                            )
test['label'] = predicted_probs
test[['id','label']].to_csv('submission.csv', index=False)