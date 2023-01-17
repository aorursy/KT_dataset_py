import numpy as np
import pandas as pd 
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
import gc
df = pd.read_csv('../input/openimagevallabel/validation-annotations-human-imagelabels.csv', usecols=[0,2,3])
df = df[df.Confidence == 1]
classes = np.array(['/m/01g317', '/m/09j2d', '/m/04yx4', '/m/0dzct', '/m/07j7r', '/m/05s2s', '/m/03bt1vf', '/m/07yv9', '/m/0cgh4', '/m/01prls', '/m/09j5n', '/m/0jbk', '/m/0k4j', '/m/05r655', '/m/02wbm', '/m/0c9ph5', '/m/083wq', '/m/0c_jw', '/m/03jm5', '/m/0d4v4'])
li = []
for i in classes:
    li.append(df[df.LabelName == i])
df = pd.concat(li).sample(frac=1).reset_index(drop=True)
del li
gc.collect()
df.head()
labels = df.LabelName.tolist()
Imageid = df.ImageID.values
print(len(df))
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Dense, Conv2D, PReLU, BatchNormalization, MaxPooling2D, Dropout, Flatten
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras.models import load_model, Sequential
from tqdm import tqdm_notebook
from tqdm import tqdm
from keras.optimizers import Adam
df.head(),gc.collect()
X_train = [np.array(load_img('../input/open-image-val/validation/validation/{}.jpg'.format(i),target_size=(100,100), grayscale=True))/255 for i in tqdm(Imageid[10000:20000])]
X_Val = [np.array(load_img('../input/open-image-val/validation/validation/{}.jpg'.format(i),target_size=(100,100), grayscale=True))/255 for i in tqdm(Imageid[:2000])]
#Y_train = labels[10000:20000]
#Y_Val = labels[:2000]
classes = classes.tolist()
Y_train, Y_Val = [], []
for i in tqdm(labels[10000:20000]):
    temp = np.zeros(20)
    temp[classes.index(i)] = 1
    Y_train.append(temp)
    del temp
for i in tqdm(labels[:2000]):
    temp = np.zeros(20)
    temp[classes.index(i)] = 1
    Y_Val.append(temp)
    del temp
Y_train = np.array(Y_train)
Y_Val = np.array(Y_Val)
gc.collect(), Y_train[0]
nn = Sequential()
nn.add(BatchNormalization(input_shape=(100, 100, 1)))
nn.add(Conv2D(4, kernel_size=(2,2), strides=(1,1)))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(0.25))
nn.add(Conv2D(8, kernel_size=(2,2), strides=(1,1)))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(0.25))
nn.add(Conv2D(16, kernel_size=(2,2), strides=(2,2)))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(0.25))
nn.add(Conv2D(32, kernel_size=(2,2), strides=(1,1)))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(0.25))
nn.add(Conv2D(32, kernel_size=(2,2), strides=(2,2)))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(0.25))
nn.add(Conv2D(32, kernel_size=(2,2), strides=(2,2)))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(0.25))
nn.add(Flatten())
nn.add(Dense(2048))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(0.25))
nn.add(Dense(1024))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(0.25))
nn.add(Dense(512))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(0.25))
nn.add(Dense(128))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(0.25))
nn.add(Dense(50))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(0.25))
nn.add(Dense(25))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(0.25))
nn.add(Dense(20, activation='softmax'))
nn.compile(loss=keras.losses.categorical_crossentropy, metrics=['accuracy'], optimizer='adam')
nn.summary()
X_train = np.array(X_train).reshape((10000,100,100,1))
X_Val = np.array(X_Val).reshape((2000,100,100,1))
gc.collect()
nn.fit(X_train, Y_train, validation_data=(X_Val,Y_Val), batch_size=100, epochs=50, verbose=2)
del X_train, Y_train, X_Val, Y_Val, df
gc.collect()
df = pd.read_csv('../input/inclusive-images-challenge/stage_1_sample_submission.csv', usecols=[0])
im = df.image_id.tolist()
df.head()
X_test = [np.array(load_img('../input/inclusive-images-challenge/stage_1_test_images/{}.jpg'.format(i),target_size=(100,100), grayscale=True))/255 for i in tqdm_notebook(im)]
X_test = np.array(X_test).reshape((32580,100,100,1))
pre = nn.predict(X_test).argsort(1)[:,:5]
del X_test
gc.collect()
p = []
for it in tqdm(pre):
    p.append(' '.join([classes[int(i)] for i in it]))
df['labels'] = p
df.head()
df.to_csv('sub.csv', index=False)