# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!unzip /kaggle/input/denoising-dirty-documents/train.zip

!unzip /kaggle/input/denoising-dirty-documents/test.zip

!unzip /kaggle/input/denoising-dirty-documents/train_cleaned.zip

!unzip /kaggle/input/denoising-dirty-documents/sampleSubmission.csv.zip
import numpy as np

from keras.layers import Input, Dense,Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization

from keras.models import Model,Sequential

import matplotlib.pyplot as plt

import cv2

from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
plt.figure(figsize=(20, 2))

for root, dirs, files in os.walk('/kaggle/working/train'):

    for i in range(5):

        ax = plt.subplot(1, 5, i+1)

        img = cv2.imread(os.path.join(root,files[i]))

        resized = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)

        plt.imshow(resized)

plt.show()
plt.figure(figsize=(20, 2))

for root, dirs, files in os.walk('/kaggle/working/train_cleaned'):

    for i in range(5):

        ax = plt.subplot(1, 5, i+1)

        img = cv2.imread(os.path.join(root,files[i]))

        resized = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)

        plt.imshow(resized)

plt.show()
# Load train and train_cleaned data

train_data = []

train_data_cleaned = []

train_path = '/kaggle/working/train'

train_cleaned_path = '/kaggle/working/train_cleaned'







for filename in os.listdir(train_path):

    train_img = cv2.imread(os.path.join(train_path,filename))

    train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    train_img = cv2.resize(train_img,(540, 258),interpolation = cv2.INTER_AREA)

    train_img = train_img.astype('float32')

    train_img = train_img/255.0

    train_data.append(train_img)

    train_cleaned_img = cv2.imread(os.path.join(train_cleaned_path,filename))

    train_cleaned_img = cv2.cvtColor(train_cleaned_img, cv2.COLOR_BGR2GRAY)

    train_cleaned_img = cv2.resize(train_cleaned_img,(540, 258),interpolation = cv2.INTER_AREA)

    train_cleaned_img = train_cleaned_img.astype('float32')

    train_cleaned_img = train_cleaned_img/255.0

    train_data_cleaned.append(train_cleaned_img)
# Let's stack the images

train_data = np.stack(train_data)

train_data_cleaned = np.stack(train_data_cleaned)



# Reshaping the data for model

train_data = train_data.reshape(train_data.shape[0],train_data.shape[1],train_data.shape[2],1)

train_data_cleaned = train_data_cleaned.reshape(train_data_cleaned.shape[0],train_data_cleaned.shape[1],train_data_cleaned.shape[2],1)







x_train,x_val,y_train,y_val = train_test_split(train_data,train_data_cleaned,test_size=0.2)

# Define the model

input_img = Input(shape=(258,540,1))

encoder = Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='elu')(input_img)

encoder = MaxPooling2D((2,2))(encoder)

decoder = Conv2D(64,kernel_size=(3,3),padding='same',activation='elu')(encoder)

decoder = UpSampling2D((2,2))(decoder)

decoder = Conv2D(1,kernel_size=(3,3),padding='same',activation='sigmoid')(decoder)

autoencoder = Model(input_img,decoder)

autoencoder.compile(loss='binary_crossentropy',optimizer='adam',metrics=['mse'])



autoencoder.summary()

early_stopping = EarlyStopping(monitor='val_loss',min_delta=0,patience=5,verbose=1, mode='auto')
history = autoencoder.fit(x_train,y_train,epochs=100,batch_size=20,validation_data=(x_val,y_val),callbacks=[early_stopping])
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.xlabel('Epochs')

plt.ylabel('Model Loss')

plt.legend(['Train', 'Test'])

plt.show()
preds = autoencoder.predict(x_val)
preds_0 = preds[0].reshape(preds.shape[1],preds.shape[2])

x_val_0 = x_val[0].reshape(x_val.shape[1],x_val.shape[2])

plt.imshow(preds_0,cmap='gray')
preds_1= preds[1].reshape(preds.shape[1],preds.shape[2])

x_val_1 = x_val[1].reshape(x_val.shape[1],x_val.shape[2])

plt.imshow(preds_1,cmap='gray')
def load_test(path):

    test_data= []

    test_keys= []

    for filename in os.listdir(path):

        test_key = filename.split('.')[0]

        img = cv2.imread(os.path.join(path,filename))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img,(540, 258),interpolation = cv2.INTER_AREA)

        img = img.astype('float32')

        img = img/255.0

        test_data.append(img)

        test_keys.append(test_key)

    return test_data,test_keys



test_path = '/kaggle/working/test'

test_data,test_keys = load_test(test_path)

test_data = np.stack(test_data)

test_data = test_data.reshape(test_data.shape[0],test_data.shape[1],test_data.shape[2],1)
test_preds = autoencoder.predict(test_data)
test_preds_1= test_preds[1].reshape(test_preds.shape[1],test_preds.shape[2])

test_data_1 = test_data[1].reshape(test_data.shape[1],test_data.shape[2])

plt.imshow(test_data_1,cmap='gray')
plt.imshow(test_preds_1,cmap='gray')