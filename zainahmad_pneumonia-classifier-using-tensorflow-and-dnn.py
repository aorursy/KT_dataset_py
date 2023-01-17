# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Input , Dense , Flatten , Dropout , BatchNormalization
from tensorflow.keras.layers import Conv2D , SeparableConv1D , ReLU , Activation , MaxPool2D
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from PIL import Image

print("tensorflow version :" ,tf.__version__)
train_norm = "/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL"
train_pnuem = "/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA"
test_norm = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL"
test_pnuem = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA"
val_norm = "/kaggle/input/chest-xray-pneumonia/chest_xray/val/NORMAL"
val_pnuem = "/kaggle/input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA"
train = "/kaggle/input/chest-xray-pneumonia/chest_xray/train"
test = "/kaggle/input/chest-xray-pneumonia/chest_xray/test"
val = "/kaggle/input/chest-xray-pneumonia/chest_xray/val"
def plot_img(item_dir , num_img = 3):
    all_dir = os.listdir(item_dir)
    files = [os.path.join(item_dir , file) for file in all_dir][:num_img]
    
    plt.figure(figsize=(20 , 12))
    for index , img_p in enumerate(files):
        plt.subplot(2 ,3, index +1)
        img = plt.imread(img_p)
        plt.imshow(img , cmap='gray')
    plt.tight_layout()
print("CHEST X-RAY OF NORMAL PERSON")
plot_img(train_norm)
print("CHEST X-RAY OF A PNEUMONIA PATIENT")
plot_img(train_pnuem)
img_dims = 256
BATCH_SIZE = 32
EPOCHS = 20
def load_data(img_dims , batch_size):
    train_loader = ImageDataGenerator(rescale= 1./255 , zoom_range=0.3 , vertical_flip=True)
    test_loader = ImageDataGenerator(rescale= 1./ 255 , zoom_range = 0.3)
    val_loader = ImageDataGenerator(rescale= 1./255)
    
    train_data_gen = train_loader.flow_from_directory(directory=train , 
                                                  batch_size=batch_size ,
                                                  shuffle = True ,
                                                 target_size= (img_dims , img_dims),
                                                  class_mode='binary')
    test_data_gen = test_loader.flow_from_directory(directory=test ,
                                                batch_size= batch_size ,
                                                shuffle = True ,
                                                target_size= (img_dims , img_dims) ,
                                                class_mode= 'binary')
    val_data_gen = val_loader.flow_from_directory(directory=val ,
                                              batch_size=batch_size ,
                                              shuffle= True ,
                                              target_size = (img_dims , img_dims) ,
                                              class_mode= 'binary')
    return train_data_gen , test_data_gen , val_data_gen
train_data , test_data , val_data = load_data(img_dims=img_dims , batch_size=BATCH_SIZE)
model = Sequential()
# layer 1
model.add(Conv2D(filters= 16 , kernel_size = (3,3) , activation = 'relu' ,input_shape = (img_dims , img_dims , 3) , padding = 'same' ,))
model.add(MaxPool2D(pool_size = (2, 2)))
# layer 2
model.add(Conv2D(filters = 32 , kernel_size =(3,3) , activation = 'relu' , padding = 'same'))
model.add(MaxPool2D(pool_size = (2,2)))
# layer 3
model.add(Conv2D(filters = 64 , kernel_size =(3,3) , activation = 'relu' , padding = 'same'))
model.add(MaxPool2D(pool_size = (2,2)))
# layer 4 FLATTEN
model.add(Flatten())
# layer 5 FC
model.add(Dense(512 , activation = 'relu'))
# layer 6 FC
model.add(Dense(256 , activation = 'relu'))
# output
model.add(Dense(1 , activation = 'sigmoid'))
model.summary()
model.compile(optimizer= 'adam' ,
              loss = 'binary_crossentropy' ,
              metrics= ['accuracy'])
# checkpoint
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="best_weights.hdf5", save_best_only = True ,save_weights_only=True , verbose=1)

model = Sequential()
# layer 1
model.add(Conv2D(filters= 16 , kernel_size = (3,3) , activation = 'relu' ,input_shape = (img_dims , img_dims , 3) , padding = 'same' ,))
model.add(MaxPool2D(pool_size = (2, 2)))
# layer 2
model.add(Conv2D(filters = 32 , kernel_size =(3,3) , activation = 'relu' , padding = 'same'))
model.add(MaxPool2D(pool_size = (2,2)))
# layer 3
model.add(Conv2D(filters = 64 , kernel_size =(3,3) , activation = 'relu' , padding = 'same'))
model.add(MaxPool2D(pool_size = (2,2)))
# layer 4 FLATTEN
model.add(Flatten())
# layer 5 FC
model.add(Dense(512 , activation = 'relu'))
# DROPOUT 1 
model.add(Dropout(rate= 0.7))
# layer 6 FC
model.add(Dense(256 , activation = 'relu'))
# DROPOUT 2
model.add(Dropout(rate = 0.5))
#layer 7 FC
model.add(Dense(128 , activation = 'relu'))
# DROPOUT 3
model.add(Dropout(rate = 0.3))
# output
model.add(Dense(1 , activation = 'sigmoid'))
model.summary()
model.compile(optimizer= 'adam' ,
              loss = 'binary_crossentropy' ,
              metrics= ['accuracy'])
history  = model.fit(train_data ,
                     epochs=EPOCHS , 
                     validation_data=test_data , 
                     callbacks=[cp_callback])


