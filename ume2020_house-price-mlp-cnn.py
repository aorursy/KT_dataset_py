# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import datetime

import random

import glob

import cv2

import os

import tensorflow as tf

import keras.backend as K

from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import plot_model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import BatchNormalization,Activation,Dropout,Dense

from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau



from tensorflow.keras.applications import VGG16



from sklearn.preprocessing import StandardScaler  

from sklearn.model_selection import KFold



from keras.layers import Input

from keras.models import Model



from tensorflow.keras.preprocessing.image import ImageDataGenerator



import matplotlib.pyplot as plt

%matplotlib inline



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)
#数値データ読込

train = pd.read_csv('/kaggle/input/4th-datarobot-ai-academy-deep-learning/train.csv')

test = pd.read_csv('/kaggle/input/4th-datarobot-ai-academy-deep-learning/test.csv')

display(train.shape)

display(train.head())
# 正規化

scaler = StandardScaler()

train[['bedrooms','bathrooms','area','zipcode']] = scaler.fit_transform(train[['bedrooms','bathrooms','area','zipcode']])

test[['bedrooms','bathrooms','area','zipcode']] = scaler.fit_transform(test[['bedrooms','bathrooms','area','zipcode']])
train_x,train_y = train[['bedrooms','bathrooms','area','zipcode']].values,train['price'].values

test_x = test[['bedrooms','bathrooms','area','zipcode']].values
#画像読込＋結合　[0,0]家、[0,1]バス、[1,0]ベット、[0,0]キッチン

def load_images(df,inputPath,size):

    images = []

    for i in df['id']:

        basePath = os.path.sep.join([inputPath, "{}_*".format(i)])

        housePaths = sorted(list(glob.glob(basePath)))

        print(housePaths)

        imgs=[]

        image4 = np.zeros((64, 64, 3), dtype="uint8")

        for i,name in enumerate(housePaths):

            image = cv2.imread(name)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(image, (size, size))

            imgs.append(image)

        image4=np.vstack((np.hstack((imgs[2],imgs[0])), np.hstack((imgs[3],imgs[1]))))

        images.append(image4)

    return np.array(images) / 255.0



# load train images

inputPath = '/kaggle/input/4th-datarobot-ai-academy-deep-learning/images/train_images/'

size = 32

train_images = load_images(train,inputPath,size)

inputPath = '/kaggle/input/4th-datarobot-ai-academy-deep-learning/images/test_images/'

test_images = load_images(test,inputPath,size)



display(train_images.shape)

#display(train_images[0][0][0])
#MLP

def mlp(num_cols):

    

    model = Sequential()

    model.add(Dense(units=512, input_shape = (len(num_cols),), 

                    kernel_initializer='he_normal',activation='relu'))    

    model.add(Dropout(0.2))

    model.add(Dense(units=256,  kernel_initializer='he_normal',activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(units=32, kernel_initializer='he_normal', activation='relu'))     

    model.add(Dropout(0.2))

    model.add(Dense(1, activation='linear'))

    model.compile(loss='mape', optimizer='adam', metrics=['mape']) 

    return model
#vgg16

def vgg16_finetuning(inputShape):

    backbone = VGG16(weights='imagenet',

                    include_top=False,

                    input_shape=inputShape)

    

    for layer in backbone.layers[:15]:

        layer.trainable = False

    for layer in backbone.layers:

        print("{}: {}".format(layer, layer.trainable))

        

    model = Sequential(layers=backbone.layers)     

    model.add(GlobalAveragePooling2D())

    model.add(Dense(units=256, activation='relu',kernel_initializer='he_normal'))  

    model.add(Dense(units=32, activation='relu',kernel_initializer='he_normal'))    

    model.add(Dense(units=1, activation='linear'))

    

    model.compile(loss='mape', optimizer='adamax', metrics=['mape']) 

    model.summary()

    return model
#MLP api用

def mlp_api(num_cols): 

    inputs = Input(len(num_cols),) 

    x=Dense(units=512, kernel_initializer='he_normal',activation='relu')(inputs)  

    x=Dropout(0.2)(x)

    x=Dense(units=256,  kernel_initializer='he_normal',activation='relu')(x)

    x=Dropout(0.2)(x)

    x=Dense(units=32, kernel_initializer='he_normal', activation='relu')(x)

    x=Dropout(0.2)(x)



    return Model(inputs=inputs, outputs=x)
#cnn api用

def create_cnn_api(inputShape,dropout_rate,kernel_s):

    inputs = Input(inputShape)

    x=Conv2D(filters=32, kernel_size=(kernel_s, kernel_s), strides=(1, 1), padding='same',

                     activation='relu', kernel_initializer='he_normal')(inputs)

    x=MaxPooling2D(pool_size=(2, 2))(x)

    x=BatchNormalization()(x)

    x=Dropout(dropout_rate)(x)



    x=Conv2D(filters=64, kernel_size=(kernel_s, kernel_s), strides=(1, 1), padding='same', 

                     activation='relu', kernel_initializer='he_normal')(x)

    x=MaxPooling2D(pool_size=(2, 2))(x)

    x=BatchNormalization()(x)

    x=Dropout(dropout_rate)(x)

     

    x=Flatten()(x)

    

    x=Dense(units=256, activation='relu',kernel_initializer='he_normal')(x)

    x=Dense(units=32, activation='relu',kernel_initializer='he_normal')(x)



    return Model(inputs=inputs, outputs=x)
#MAPE算出

def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true- y_pred) / y_true)) * 100
#設定値

k_splits=5

epoch=200

batch_size=32

numcols=['bedrooms','bathrooms','area','zipcode']

inputShape = (size*2, size*2, 3)



# デザイン

datagen = ImageDataGenerator(horizontal_flip=True,

                             vertical_flip=True,

                            rotation_range=90,

                             width_shift_range=0.2,

                             height_shift_range=0.2,

                             )
seed_everything(2020)
#MLPモデル



#セッションのクリア

K.clear_session()

# callback parameter

filepath = "2020_1_mlp_best_model.hdf5" 

es = EarlyStopping(patience=5, mode='min', verbose=1) 

checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min')



# 訓練実行

kfold = KFold(n_splits=k_splits, shuffle=True)



for t, v in kfold.split(train_y):

 model= mlp(numcols)

 history = model.fit([train_x[t]], train_y[t], validation_data=([train_x[v]], train_y[v]),epochs=epoch, batch_size=batch_size,

 callbacks=[es, checkpoint, reduce_lr_loss])

# load best model weights

model.load_weights(filepath)
# load best model weights

model.load_weights(filepath)

# 評価 train_images[v], train_y[v]

for t, v in kfold.split(train_y):

 valid_pred = model.predict([train_x[v]], batch_size=32).reshape(-1)

 mape_score = mean_absolute_percentage_error(train_y[v], valid_pred)

 print (mape_score)
#Transfer Learning、Data Augmentation

#セッションのクリア

K.clear_session()

import keras

# callback parameter

filepath = "2020_data_aug_best_model.hdf5" 

es = EarlyStopping(patience=5, mode='min', verbose=1) 

checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min')



# 訓練実行

kfold = KFold(n_splits=k_splits, shuffle=True)



for t, v in kfold.split(train_y):

 model = vgg16_finetuning(inputShape)

 datagen.fit(train_images[t],augment=True)

 train_datagen = datagen.flow(train_images[t], train_y[t], batch_size=batch_size, shuffle=True)

 history = model.fit(train_datagen, validation_data=(train_images[v], train_y[v]),

     steps_per_epoch=len(train_images[t]) / batch_size, epochs=epoch,                

     callbacks=[es, checkpoint, reduce_lr_loss])
# load best model weights

model.load_weights(filepath)



# 評価 train_images[v], train_y[v]

for t, v in kfold.split(train_y):

 valid_pred = model.predict(train_images[v], batch_size=32).reshape(-1)

 mape_score = mean_absolute_percentage_error(train_y[v], valid_pred)

 print (mape_score)
#api

#セッションのクリア

K.clear_session()

import keras

# callback parameter

filepath = "2020_api_best_model.hdf5" 



es = EarlyStopping(patience=5, mode='min', verbose=1) 

checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min')



# 訓練実行

kfold = KFold(n_splits=k_splits, shuffle=True)



for t, v in kfold.split(train_y):

 x= mlp_api(numcols)

 y= create_cnn_api(inputShape,0.5,3)

 #y= vgg16_finetuning(inputShape)



 combined = keras.layers.concatenate([x.output, y.output],axis=-1)

 z=Dense(1, activation='linear')(combined)

 model = Model(inputs=[x.input, y.input], outputs=z)

 model.compile(loss='mape', optimizer='adam', metrics=['mape']) 



 history = model.fit([train_x[t],train_images[t]], train_y[t], validation_data=([train_x[v],train_images[v]], train_y[v]),epochs=epoch, batch_size=batch_size,

 callbacks=[es, checkpoint, reduce_lr_loss])
# load best model weights

model.load_weights(filepath)



# 評価 train_images[v], train_y[v]

for t, v in kfold.split(train_y):

 valid_pred = model.predict([train_x[v],train_images[v]], batch_size=32).reshape(-1)

 mape_score = mean_absolute_percentage_error(train_y[v], valid_pred)

 print (mape_score)
seed_everything(1010)
#MLPモデル



#セッションのクリア

K.clear_session()

# callback parameter

filepath = "1010_1_mlp_best_model.hdf5" 

es = EarlyStopping(patience=5, mode='min', verbose=1) 

checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min')



# 訓練実行

kfold = KFold(n_splits=k_splits, shuffle=True)



for t, v in kfold.split(train_y):

 model= mlp(numcols)

 history = model.fit([train_x[t]], train_y[t], validation_data=([train_x[v]], train_y[v]),epochs=epoch, batch_size=batch_size,

 callbacks=[es, checkpoint, reduce_lr_loss])

# load best model weights

model.load_weights(filepath)
# load best model weights

model.load_weights(filepath)

# 評価 train_images[v], train_y[v]

for t, v in kfold.split(train_y):

 valid_pred = model.predict([train_x[v]], batch_size=32).reshape(-1)

 mape_score = mean_absolute_percentage_error(train_y[v], valid_pred)

 print (mape_score)
#Transfer Learning、Data Augmentation

#セッションのクリア

K.clear_session()

import keras

# callback parameter

filepath = "1010_data_aug_best_model.hdf5" 

es = EarlyStopping(patience=5, mode='min', verbose=1) 

checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min')



# 訓練実行

kfold = KFold(n_splits=k_splits, shuffle=True)



for t, v in kfold.split(train_y):

 model = vgg16_finetuning(inputShape)

 datagen.fit(train_images[t],augment=True)

 train_datagen = datagen.flow(train_images[t], train_y[t], batch_size=batch_size, shuffle=True)

 history = model.fit(train_datagen, validation_data=(train_images[v], train_y[v]),

     steps_per_epoch=len(train_images[t]) / batch_size, epochs=epoch,                

     callbacks=[es, checkpoint, reduce_lr_loss])
# load best model weights

model.load_weights(filepath)



# 評価 train_images[v], train_y[v]

for t, v in kfold.split(train_y):

 valid_pred = model.predict(train_images[v], batch_size=32).reshape(-1)

 mape_score = mean_absolute_percentage_error(train_y[v], valid_pred)

 print (mape_score)
#api

#セッションのクリア

K.clear_session()

import keras

# callback parameter

filepath = "1010_api_best_model.hdf5" 



es = EarlyStopping(patience=5, mode='min', verbose=1) 

checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min')



# 訓練実行

kfold = KFold(n_splits=k_splits, shuffle=True)



for t, v in kfold.split(train_y):

 x= mlp_api(numcols)

 y= create_cnn_api(inputShape,0.5,3)

 #y= vgg16_finetuning(inputShape)



 combined = keras.layers.concatenate([x.output, y.output],axis=-1)

 z=Dense(1, activation='linear')(combined)

 model = Model(inputs=[x.input, y.input], outputs=z)

 model.compile(loss='mape', optimizer='adam', metrics=['mape']) 



 history = model.fit([train_x[t],train_images[t]], train_y[t], validation_data=([train_x[v],train_images[v]], train_y[v]),epochs=epoch, batch_size=batch_size,

 callbacks=[es, checkpoint, reduce_lr_loss])
# load best model weights

model.load_weights(filepath)



# 評価 train_images[v], train_y[v]

for t, v in kfold.split(train_y):

 valid_pred = model.predict([train_x[v],train_images[v]], batch_size=32).reshape(-1)

 mape_score = mean_absolute_percentage_error(train_y[v], valid_pred)

 print (mape_score)
seed_everything(71)
#MLPモデル



#セッションのクリア

K.clear_session()

# callback parameter

filepath = "71_1_mlp_best_model.hdf5" 

es = EarlyStopping(patience=5, mode='min', verbose=1) 

checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min')



# 訓練実行

kfold = KFold(n_splits=k_splits, shuffle=True)



for t, v in kfold.split(train_y):

 model= mlp(numcols)

 history = model.fit([train_x[t]], train_y[t], validation_data=([train_x[v]], train_y[v]),epochs=epoch, batch_size=batch_size,

 callbacks=[es, checkpoint, reduce_lr_loss])

# load best model weights

model.load_weights(filepath)
# load best model weights

model.load_weights(filepath)

# 評価 train_images[v], train_y[v]

for t, v in kfold.split(train_y):

 valid_pred = model.predict([train_x[v]], batch_size=32).reshape(-1)

 mape_score = mean_absolute_percentage_error(train_y[v], valid_pred)

 print (mape_score)
#Transfer Learning、Data Augmentation

#セッションのクリア

K.clear_session()

import keras

# callback parameter

filepath = "71_data_aug_best_model.hdf5" 

es = EarlyStopping(patience=5, mode='min', verbose=1) 

checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min')



# 訓練実行

kfold = KFold(n_splits=k_splits, shuffle=True)



for t, v in kfold.split(train_y):

 model = vgg16_finetuning(inputShape)

 datagen.fit(train_images[t],augment=True)

 train_datagen = datagen.flow(train_images[t], train_y[t], batch_size=batch_size, shuffle=True)

 history = model.fit(train_datagen, validation_data=(train_images[v], train_y[v]),

     steps_per_epoch=len(train_images[t]) / batch_size, epochs=epoch,                

     callbacks=[es, checkpoint, reduce_lr_loss])
# load best model weights

model.load_weights(filepath)



# 評価 train_images[v], train_y[v]

for t, v in kfold.split(train_y):

 valid_pred = model.predict(train_images[v], batch_size=32).reshape(-1)

 mape_score = mean_absolute_percentage_error(train_y[v], valid_pred)

 print (mape_score)
#api

#セッションのクリア

K.clear_session()

import keras

# callback parameter

filepath = "71_api_best_model.hdf5" 



es = EarlyStopping(patience=5, mode='min', verbose=1) 

checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min')



# 訓練実行

kfold = KFold(n_splits=k_splits, shuffle=True)



for t, v in kfold.split(train_y):

 x= mlp_api(numcols)

 y= create_cnn_api(inputShape,0.5,3)

 #y= vgg16_finetuning(inputShape)



 combined = keras.layers.concatenate([x.output, y.output],axis=-1)

 z=Dense(1, activation='linear')(combined)

 model = Model(inputs=[x.input, y.input], outputs=z)

 model.compile(loss='mape', optimizer='adam', metrics=['mape']) 



 history = model.fit([train_x[t],train_images[t]], train_y[t], validation_data=([train_x[v],train_images[v]], train_y[v]),epochs=epoch, batch_size=batch_size,

 callbacks=[es, checkpoint, reduce_lr_loss])
# load best model weights

model.load_weights(filepath)



# 評価 train_images[v], train_y[v]

for t, v in kfold.split(train_y):

 valid_pred = model.predict([train_x[v],train_images[v]], batch_size=32).reshape(-1)

 mape_score = mean_absolute_percentage_error(train_y[v], valid_pred)

 print (mape_score)
# mlp

# load best model weights

model= mlp(numcols)

filepath = "2020_1_mlp_best_model.hdf5" 

model.load_weights(filepath)

valid_pred_mlp_2020 = model.predict(test_x, batch_size=32).reshape(-1)

# load best model weights

model= mlp(numcols)

filepath = "1010_1_mlp_best_model.hdf5" 

valid_pred_mlp_1010 = model.predict(test_x, batch_size=32).reshape(-1)

# load best model weights

model= mlp(numcols)

filepath = "71_1_mlp_best_model.hdf5" 

model.load_weights(filepath)

valid_pred_mlp_71 = model.predict(test_x, batch_size=32).reshape(-1)



#vgg16

# load best model weights

model = vgg16_finetuning(inputShape)

filepath = "2020_data_aug_best_model.hdf5" 

model.load_weights(filepath)

valid_pred_vgg16_2020 = model.predict(test_images, batch_size=32).reshape(-1)

# load best model weights

model = vgg16_finetuning(inputShape)

filepath = "1010_data_aug_best_model.hdf5" 

model.load_weights(filepath)

valid_pred_vgg16_1010 = model.predict(test_images, batch_size=32).reshape(-1)

# load best model weights

model = vgg16_finetuning(inputShape)

filepath = "71_data_aug_best_model.hdf5" 

model.load_weights(filepath)

valid_pred_vgg16_71 = model.predict(test_images, batch_size=32).reshape(-1)



#api

# load best model weights

x= mlp_api(numcols)

y= create_cnn_api(inputShape,0.5,3)

combined = keras.layers.concatenate([x.output, y.output],axis=-1)

z=Dense(1, activation='linear')(combined)

model = Model(inputs=[x.input, y.input], outputs=z)

model.compile(loss='mape', optimizer='adam', metrics=['mape'])

filepath = "2020_api_best_model.hdf5" 

model.load_weights(filepath)

valid_pred_api_2020 = model.predict([test_x,test_images], batch_size=32).reshape(-1)

# load best model weights

x= mlp_api(numcols)

y= create_cnn_api(inputShape,0.5,3)

combined = keras.layers.concatenate([x.output, y.output],axis=-1)

z=Dense(1, activation='linear')(combined)

model = Model(inputs=[x.input, y.input], outputs=z)

model.compile(loss='mape', optimizer='adam', metrics=['mape'])

filepath = "1010_api_best_model.hdf5" 

model.load_weights(filepath)

valid_pred_api_1010 = model.predict([test_x,test_images], batch_size=32).reshape(-1)

# load best model weights

x= mlp_api(numcols)

y= create_cnn_api(inputShape,0.5,3)

combined = keras.layers.concatenate([x.output, y.output],axis=-1)

z=Dense(1, activation='linear')(combined)

model = Model(inputs=[x.input, y.input], outputs=z)

model.compile(loss='mape', optimizer='adam', metrics=['mape'])

filepath = "71_api_best_model.hdf5" 

model.load_weights(filepath)

valid_pred_api_71 = model.predict([test_x,test_images], batch_size=32).reshape(-1)



submission = pd.DataFrame({

"id": test.id,

"price": (valid_pred_vgg16_2020+valid_pred_mlp_2020+valid_pred_api_2020+valid_pred_vgg16_1010+valid_pred_mlp_1010+valid_pred_api_1010+valid_pred_vgg16_71+valid_pred_mlp_71+valid_pred_api_71)/9

})

submission.to_csv('submission.csv', index=False)