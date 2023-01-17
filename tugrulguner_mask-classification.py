# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow.keras as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
#from keras.applications.inception_v3 import InceptionV3

from keras.applications import MobileNetV2, VGG19, VGG16, Xception, InceptionV3, InceptionResNetV2, DenseNet201
import cv2
import os

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/face-mask-detection-dataset/train.csv')
deneme = [1 if ((element=='face_with_mask') or  (element=='mask_surgical') or (element=='mask_colorful')) else 0 for element in data.classname]
data['Classification'] = deneme
data.drop(['x1', 'x2', 'y1', 'y2', 'classname'], axis =1, inplace=True)
data
datagrouped = data.groupby(['name'], as_index=False).sum()
datagrouped.Classification[datagrouped['Classification']>=1]=1
datagrouped
path = '/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/'
rre = cv2.imread(path+'3216.png')
plt.imshow(rre)
path = '/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/'
trainingdata = np.zeros((1,256,256,3))
k = 0
for element in datagrouped['name']:
    resim = cv2.imread(path+element)
    resim = cv2.resize(resim,(256,256), interpolation = cv2.INTER_NEAREST)
    resim = (resim-np.mean(resim))/np.std(resim)
    trainingdata = np.append(trainingdata, resim[np.newaxis,...], axis=0) 
    if k==1200:
        break
    k+=1
ag = np.delete(trainingdata,0, axis=0)
ag.shape
plt.imshow(ag[25,...])
sns.distplot(ag[25,125:225,50:150,0])
sns.distplot(ag[25,...,0])
plt.imshow(ag[25,...]*(ag[25,...]>0.6))
trainingdata[1000:1500,...].shape
#labeldata = np.zeros((1000,2))
#for el, em in enumerate(labeldata):
#    if em == 0:
#        continue
#    else:        
labeldata = datagrouped.Classification[:1201]
labeldata
input_layer = tf.Input(shape=(256, 256, 3))
#model = VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
for layer in model.layers[:-8]:
    layer.trainable = False
x = tf.layers.Conv2D(32,(3, 3), strides=(1, 1), activation='relu')(input_layer)
x = tf.layers.Conv2D(32,(3, 3), strides=(1, 1), activation='relu')(x)
#x = tf.layers.Conv2D(64,(3, 3), strides=(1, 1), activation='relu')(x)
#x = tf.layers.Conv2D(128,(3, 3), padding='same', strides=(1, 1), activation='relu')(x)
x = tf.layers.BatchNormalization()(x)
x = tf.layers.MaxPool2D((2, 2), strides=(2, 2))(x)
x = tf.layers.Dropout(0.5)(x)
x = tf.layers.Conv2D(64,(3, 3), strides=(1, 1), activation='relu')(x)
x = tf.layers.Conv2D(64,(3, 3), strides=(1, 1), activation='relu')(x)
#x = tf.layers.Conv2D(64,(3, 3), strides=(1, 1), activation='relu')(x)
x = tf.layers.BatchNormalization()(x)
#x = tf.layers.Conv2D(64,(3, 3), strides=(1, 1), activation='relu')(x)
#x = tf.layers.Conv2D(256,(3, 3), padding='same', strides=(1, 1), activation='relu')(x)
x = tf.layers.MaxPool2D((2, 2), strides=(2, 2))(x)
x = tf.layers.Conv2D(32,(3, 3), strides=(1, 1), activation='relu')(x)
x = tf.layers.Conv2D(32,(3, 3), strides=(1, 1), activation='relu')(x)
x = tf.layers.BatchNormalization()(x)
#x = tf.layers.Conv2D(64,(3, 3), strides=(1, 1), activation='relu')(x)
#x = tf.layers.Conv2D(512,(3, 3), padding='same', strides=(1, 1), activation='relu')(x)
x = tf.layers.MaxPool2D((2, 2), strides=(2, 2))(x)
#x = tf.layers.Conv2D(64,(3, 3), strides=(1, 1), activation='relu')(x)
x = tf.layers.Dropout(0.5)(x)
#x = tf.layers.Conv2D(64,(3, 3), strides=(1, 1), activation='relu')(x)
#x = tf.layers.Conv2D(64,(3, 3), strides=(1, 1), activation='relu')(x)
#x = tf.layers.BatchNormalization()(x)
#x = tf.layers.Conv2D(32,(3, 3), strides=(1, 1), activation='relu')(x)
#x = tf.layers.Conv2D(32,(3, 3), strides=(1, 1), activation='relu')(x)
#x = tf.layers.Conv2D(32,(3, 3), strides=(1, 1), activation='relu')(x)
#x = tf.layers.Conv2D(32,(3, 3), strides=(1, 1), activation='relu')(x)
#x = tf.layers.Conv2D(256,(3, 3), padding='same', strides=(1, 1), activation='relu')(x)
#x = tf.layers.MaxPool2D((2, 2), padding='same', strides=(1, 1))(x)
#x = tf.layers.BatchNormalization()(x)
#x = tf.layers.Conv2D(256, (1, 1), padding='same', strides=(1, 1), activation='relu')(x)
#x = tf.layers.Conv2D(256, (1, 1), padding='same', strides=(1, 1), activation='relu')(x)
#x = tf.layers.MaxPool2D((2, 2), padding='same', strides=(1, 1))(x)
#x = tf.layers.Conv2D(256, (1, 1), padding='same', strides=(1, 1), activation='relu')(x)
#x = tf.layers.Conv2D(256, (1, 1), padding='same', strides=(1, 1), activation='relu')(x)
#x = tf.layers.MaxPool2D((2, 2), padding='same', strides=(1, 1))(x)
#x = tf.layers.Conv2D(256, (1, 1), padding='same', strides=(1, 1), activation='relu')(x)
#x = tf.layers.Conv2D(256, (1, 1), padding='same', strides=(1, 1), activation='relu')(x)
#x = tf.layers.MaxPool2D((2, 2), strides=(2, 2))(x)
#x = tf.layers.Dropout(0.5)(x)
x = tf.layers.Conv2D(16, (1, 1), padding='same', strides=(1, 1), activation='relu')(x)
x = tf.layers.Conv2D(16, (1, 1), strides=(2, 2), activation='relu')(x)
x = tf.layers.BatchNormalization()(x)
#x = tf.layers.Conv2D(1, (3, 3), strides=(2, 2), activation='relu')(x)
#x = tf.layers.MaxPool2D((2, 2), padding='same', strides=(1, 1))(x)
#x = tf.layers.GlobalMaxPooling2D()(x)
#x = tf.layers.BatchNormalization()(x)
#x = tf.layers.Lambda(lambda x: x**2)(x)
x = tf.layers.Flatten()(x)
#x = tf.layers.GlobalAveragePooling2D()(x)
#x = tf.layers.Dense(1024, activation='relu')(x)
#x = tf.layers.Dense(2048, activation='relu')(x)
#x = tf.layers.Dense(1024, activation='relu')(x)
x = tf.layers.Dense(128, activation='relu')(x)
x = tf.layers.Dense(256, activation='relu')(x)
x = tf.layers.Dense(128, activation='relu')(x)
x = tf.layers.Dense(1, activation='sigmoid')(x)
cmodel = tf.Model(input_layer, x)
cmodel.summary()
model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
for layer in model.layers[:-8]:
    layer.trainable = False
x = tf.layers.GlobalAveragePooling2D()(model.output)
#x = tf.layers.Flatten()(model.output)
x = tf.layers.BatchNormalization()(x)
#x = tf.layers.Dropout(0.5)(x)
#x = tf.layers.Dense(2048, activation='relu')(x)
x = tf.layers.Dense(64, activation=tf.layers.PReLU())(x)
x = tf.layers.Dense(64, activation=tf.layers.PReLU())(x)
x = tf.layers.Dense(64, activation=tf.layers.PReLU())(x)
x = tf.layers.BatchNormalization()(x)
x = tf.layers.Dropout(0.5)(x)
x = tf.layers.Dense(128, activation=tf.layers.PReLU())(x)
x = tf.layers.Dense(128, activation=tf.layers.PReLU())(x)
x = tf.layers.Dense(128, activation=tf.layers.PReLU())(x)
x = tf.layers.BatchNormalization()(x)
x = tf.layers.Dropout(0.5)(x)
x = tf.layers.Dense(256, activation=tf.layers.PReLU())(x)
x = tf.layers.Dense(256, activation=tf.layers.PReLU())(x)
x = tf.layers.Dense(256, activation=tf.layers.PReLU())(x)
x = tf.layers.BatchNormalization()(x)
x = tf.layers.Dropout(0.5)(x)
x = tf.layers.Dense(512, activation=tf.layers.PReLU())(x)
x = tf.layers.BatchNormalization()(x)
x = tf.layers.Dropout(0.5)(x)
#x = tf.layers.Dense(128, activation=tf.layers.PReLU())(x)
#x = tf.layers.BatchNormalization()(x)
#x = tf.layers.Dropout(0.5)(x)
x = tf.layers.Dense(512, activation=tf.layers.PReLU())(x)
x = tf.layers.BatchNormalization()(x)
x = tf.layers.Dropout(0.5)(x)
x = tf.layers.Dense(512, activation=tf.layers.PReLU())(x)
x = tf.layers.BatchNormalization()(x)
x = tf.layers.Dropout(0.5)(x)
x = tf.layers.Dense(1024, activation=tf.layers.PReLU())(x)
x = tf.layers.Dense(1024, activation=tf.layers.PReLU())(x)
x = tf.layers.BatchNormalization()(x)
x = tf.layers.Dropout(0.5)(x)
x = tf.layers.Dense(1024, activation=tf.layers.PReLU())(x)
x = tf.layers.Dense(1024, activation=tf.layers.PReLU())(x)
x = tf.layers.BatchNormalization()(x)
x = tf.layers.Dropout(0.5)(x)
x = tf.layers.Dense(1024, activation=tf.layers.PReLU())(x)
x = tf.layers.Dense(1024, activation=tf.layers.PReLU())(x)
x = tf.layers.BatchNormalization()(x)
x = tf.layers.Dropout(0.5)(x)
x = tf.layers.Dense(512, activation=tf.layers.PReLU())(x)
x = tf.layers.BatchNormalization()(x)
x = tf.layers.Dropout(0.5)(x)
x = tf.layers.Dense(512, activation=tf.layers.PReLU())(x)
x = tf.layers.BatchNormalization()(x)
x = tf.layers.Dropout(0.5)(x)
#x = tf.layers.Dense(128, activation=tf.layers.PReLU())(x)
#x = tf.layers.BatchNormalization()(x)
#x = tf.layers.Dropout(0.5)(x)
x = tf.layers.Dense(256, activation=tf.layers.PReLU())(x)
x = tf.layers.BatchNormalization()(x)
x = tf.layers.Dropout(0.5)(x)
x = tf.layers.Dense(256, activation=tf.layers.PReLU())(x)
x = tf.layers.BatchNormalization()(x)
x = tf.layers.Dropout(0.5)(x)
x = tf.layers.Dense(128, activation=tf.layers.PReLU())(x)
x = tf.layers.Dense(128, activation=tf.layers.PReLU())(x)
x = tf.layers.Dense(128, activation=tf.layers.PReLU())(x)
x = tf.layers.BatchNormalization()(x)
x = tf.layers.Dropout(0.5)(x)
x = tf.layers.Dense(64, activation=tf.layers.PReLU())(x)
x = tf.layers.Dense(64, activation=tf.layers.PReLU())(x)
x = tf.layers.Dense(64, activation=tf.layers.PReLU())(x)
x = tf.layers.BatchNormalization()(x)
x = tf.layers.Dropout(0.5)(x)
#x = tf.layers.Dropout(0.5)(x)
#x = tf.layers.Dropout(0.8)(x)
#x = tf.layers.BatchNormalization()(x)
#x = tf.layers.Dropout(0.5)(x)
#x = tf.layers.Dense(1024, activation='relu')(x)
#x = tf.layers.BatchNormalization()(x)
#x = tf.layers.Dropout(0.5)(x)
#x = tf.layers.ActivityRegularization()(x)
x = tf.layers.Dense(1, activation='sigmoid')(x)
m = tf.Model(inputs=model.input, outputs=x)

m.summary()
m.compile(loss=tf.losses.BinaryCrossentropy(), optimizer='nadam', metrics=["accuracy"])
m.fit(ag[0:800,...], labeldata[0:800], validation_data = (ag[800:1000,...],labeldata[800:1000]), epochs=500, batch_size=100)
test_loss, test_acc = m.evaluate(ag[1000:1200,...],labeldata[1000:1200])

labeldata[1200:1210]
ag[1000:1200].shape
m.save('/kaggle/working/maskdetection.h5')
bb = tf.models.load_model("/kaggle/working/maskdetection.h5")
bb.predict(ag[1200:1210])
hh = [1,2,3,4,5]
hh[:-3]
