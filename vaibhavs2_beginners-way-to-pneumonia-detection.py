# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('dark_background')
import os
import inspect

import imblearn
from tensorflow import keras
import gc # library to free up memory usage
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Reading the input data.

# Train Data
directory = r'../input/chest-xray-pneumonia/chest_xray/chest_xray/train'
train_ls  = []
for folder in os.listdir(directory):
    if folder == 'NORMAL':
        for i in glob.glob(os.path.join(directory,folder)+'//*'):
            train_ls.append((cv2.resize(cv2.imread(i,cv2.IMREAD_GRAYSCALE),(224,224))/255.0,0))
    if folder == 'PNEUMONIA':
        for i in glob.glob(os.path.join(directory,folder)+'//*'):
            train_ls.append((cv2.resize(cv2.imread(i,cv2.IMREAD_GRAYSCALE),(224,224))/255.0,1))
        
train_df = pd.DataFrame(train_ls,columns=['Image','Label'])
train_ls.clear()

# Test Data
directory = r'../input/chest-xray-pneumonia/chest_xray/chest_xray/test'
test_ls  = []
for folder in os.listdir(directory):
    if folder == 'NORMAL':
        for i in glob.glob(os.path.join(directory,folder)+'//*'):
            test_ls.append((cv2.resize(cv2.imread(i,cv2.IMREAD_GRAYSCALE),(224,224))/255.0,0))
    if folder == 'PNEUMONIA':
        for i in glob.glob(os.path.join(directory,folder)+'//*'):
            test_ls.append((cv2.resize(cv2.imread(i,cv2.IMREAD_GRAYSCALE),(224,224))/255.0,1))
        
test_df = pd.DataFrame(test_ls,columns=['Image','Label'])
test_ls.clear()

# Validation Data
directory = r'../input/chest-xray-pneumonia/chest_xray/chest_xray/val'
val_ls  = []
for folder in os.listdir(directory):
    if folder == 'NORMAL':
        for i in glob.glob(os.path.join(directory,folder)+'//*'):
            val_ls.append((cv2.resize(cv2.imread(i,cv2.IMREAD_GRAYSCALE),(224,224))/255.0,0))
    if folder == 'PNEUMONIA':
        for i in glob.glob(os.path.join(directory,folder)+'//*'):
            val_ls.append((cv2.resize(cv2.imread(i,cv2.IMREAD_GRAYSCALE),(224,224))/255.0,1))
        
val_df = pd.DataFrame(val_ls,columns=['Image','Label'])
val_ls.clear()
# Lets have a look at our images for both normal and pneumonia
normal_sample = train_df[train_df['Label']==0][:5]
pneumonia_sample = train_df[train_df['Label']==1][:5]

fig, ax = plt.subplots(1,5, figsize=(15,15))
for i in range(0,5):
    ax[i].set_title('Normal')
    ax[i].imshow(normal_sample.iloc[i,0],cmap='gray')

fig, ax = plt.subplots(1,5, figsize=(15,15))
for j in range(0,5):
    ax[j].set_title('Pneumonia')
    ax[j].imshow(pneumonia_sample.iloc[j,0],cmap='gray')
sns.countplot(train_df['Label'])
plt.show()
print(train_df['Label'].value_counts())
# Creating Train and Test data.
X_train = np.array(train_df.drop('Label',axis=1).iloc[:,0])
X_train = np.array([x.reshape(224,224,1) for x in X_train])
y_train = np.array(train_df['Label'])
# Creating Train and Test data.
X_test = np.array(test_df.drop('Label',axis=1).iloc[:,0])
X_test = np.array([x.reshape(224,224,1) for x in X_test])
y_test = np.array(test_df['Label'])
# Creating Train and Test data.
X_val = np.array(val_df.drop('Label',axis=1).iloc[:,0])
X_val = np.array([x.reshape(224,224,1) for x in X_val])
y_val = np.array(val_df['Label'])
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
class_weights = dict(enumerate(class_weights))
class_weights
# Oversampling techniques require our data to be in 2D instead of 4D
# Converting data from 4D to 2D
# X_train_2D  = np.array(i.reshape(X_train.shape[0],X_train[i].shape[]) for i in X_train)
X_train.shape
from imblearn.over_sampling import RandomOverSampler
oversampler = RandomOverSampler(sampling_strategy=1)
X_train_res,y_train_res = oversampler.fit_sample(X_train.reshape(X_train.shape[0],(X_train.shape[1]*X_train.shape[2])),y_train)
sns.countplot(y_train_res)
plt.show()
print(pd.Series(y_train_res).value_counts())
from imblearn.under_sampling import RandomUnderSampler
undersampler = RandomUnderSampler(sampling_strategy=1)
X_train_rus,y_train_rus = undersampler.fit_sample(X_train.reshape(X_train.shape[0],(X_train.shape[1]*X_train.shape[2])),y_train)
sns.countplot(y_train_rus)
plt.show()
print(pd.Series(y_train_rus).value_counts())
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_train_sm, y_train_sm = smote.fit_sample(X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2]),y_train)
sns.countplot(y_train_sm)
plt.show()
print(pd.Series(y_train_sm).value_counts())
# Since RAM usage has gone high let us see which variables are actually using more memory and free up some space
# import sys

# local_vars = list(locals().items())
# for var, obj in local_vars:
#     print(var, sys.getsizeof(obj))
X_train.shape
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=10,
                            height_shift_range=0.1,
                            width_shift_range=0.1,
                            zoom_range=0.1,
                            vertical_flip=True,
                            horizontal_flip=True)
from kerastuner import HyperModel
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from kerastuner.tuners import RandomSearch
from keras import regularizers
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau
from keras.layers import LeakyReLU
model = Sequential()
model.add(Conv2D(input_shape=(224,224,1),filters = 32, kernel_size=(3,3)))
model.add(LeakyReLU())
model.add(MaxPool2D(pool_size=(2,2),strides=2))
model.add(Flatten())
model.add(Dense(units = 32))
model.add(LeakyReLU())
model.add(Dropout(0.3,trainable=True))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.001) , metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.0001)
history = model.fit(datagen.flow(X_train_res.reshape(-1,224,224,1),y_train_res,batch_size=128),epochs=5,validation_data=(X_val.reshape(-1,224,224,1),y_val),callbacks=[reduce_lr])
plt.plot(history.epoch,history.history['loss'],'r')
plt.plot(history.epoch,history.history['val_loss'],'b')
plt.legend()
plt.show()
from sklearn.metrics import confusion_matrix
y_pred = model.predict_classes(X_test.reshape(-1,224,224,1))
cfm = confusion_matrix(y_test,y_pred)
sns.heatmap(cfm,annot=True,fmt='g')
plt.show()
import sys
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                         key= lambda x: -x[1])[:10]:
    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
del history, model
del train_ls, test_ls, val_ls
del train_df, test_df, val_df
del X_train,y_train
del X_train_rus, y_train_rus
del y_pred
del reduce_lr
gc.collect()
