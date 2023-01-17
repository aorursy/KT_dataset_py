# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import bq_helper

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

#hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data",dataset_name = "boneage-training-dataset")
#hacker_news.list_tables()

# Any results you write to the current directory are saved as output.
#dataset = pd.read_csv("../input/boneage-training-dataset.csv")
#dataset = pd.read_csv("../input/boneage-training-dataset.zip")

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import os, sys
from IPython.display import display
from IPython.display import Image as _Imgdis
from PIL import Image
import numpy as np
from time import time
from time import sleep

folder = "../input/boneage-training-dataset/boneage-training-dataset"
folder_test = "../input/boneage-test-dataset/boneage-test-dataset"
#folder = "../input/regression_sample"

onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
onlyfiles_test = [f for f in os.listdir(folder_test) if os.path.isfile(os.path.join(folder_test, f))]

print("Working with {0} images".format(len(onlyfiles)))
print("Image examples: ")

for i in range(40, 42):
    print(onlyfiles[i])
    display(_Imgdis(filename=folder + "/" + onlyfiles[i], width=240, height=320))
    
    
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

train_files = []
test_files = []
i=0

for _file in onlyfiles:
    train_files.append(_file)
    
for _file in onlyfiles_test:
    test_files.append(_file)
    #print(train_files)
    #label_in_file = _file.find("_")
    #y_train.append(int(_file[0:label_in_file]))
print("Files in train_files: %d" % len(train_files))
print("Files in test_files: %d" % len(test_files))
#train_files[0]
#print(train_files[0:])
img_df = pd.DataFrame(data = train_files,  # This converts Array to Dataframe, with Index and Column names
                  index=None,
                  columns = None)
img_df_test = pd.DataFrame(data = test_files,  # This converts Array to Dataframe, with Index and Column names
                  index=None,
                  columns = None)

csv_df = pd.read_csv("../input/boneage-training-dataset.csv")
csv_df_test = pd.read_csv("../input/boneage-test-dataset.csv")

df_train = pd.concat([img_df,csv_df],axis = 1)  # Join two Dataframes, Finally
df_test = pd.concat([img_df_test,csv_df_test],axis = 1)

img_df = []
img_df_test = []
#pd.join(img_df,csv_df)
#img_df
#csv_df
df_train = df_train.rename(index=str, columns={0: "file"}) #Change name of Column from 0 to FIle
df_test = df_test.rename(index=str, columns={0: "file"})

df_y = df_train[['boneage']].copy()

df_train = df_train.drop(columns = ['boneage'],axis = 1)  # Dropped Y values from columns

#df_train
#df_test

image_width = 320
image_height = 240
ratio = 4

image_width = int(image_width / ratio)
image_height = int(image_height / ratio)

channels = 3
nb_classes = 1

dataset = np.ndarray(shape=(len(df_train), channels, image_height, image_width),dtype=np.float32)
dataset_test = np.ndarray(shape=(len(df_test), channels, image_height, image_width),dtype=np.float32)

# dataset = dataset[0:500,:]
# dataset_test = dataset_test
print("Dataset Train Image                "+str(dataset.shape))
print("Dataset Test Image                 "+str(dataset_test.shape))
print("Dataframe Train Values             "+str(df_train.shape))
print("Dataframe Test Values              "+str(df_test.shape))
print("Dataset Y values                   "+str(df_y.shape))

# dataset = dataset[0:500,:]
# df_train = df_train.iloc[0:500,:]
# df_y = df_y.iloc[0:500,:]

# dataset = dataset[0:5000,:]
# df_train = df_train.iloc[0:5000,:]
# df_y = df_y.iloc[0:5000,:]

print("Dataset Train Image                "+str(dataset.shape))
print("Dataset Test Image                 "+str(dataset_test.shape))
print("Dataframe Train Values             "+str(df_train.shape))
print("Dataframe Test Values              "+str(df_test.shape))
print("Dataset Y values                   "+str(df_y.shape))

df_train.iloc[0:500,:].shape
dataset.shape[0]
i = 0
#print(folder + "/" + df_train['file'])
for _file in df_train['file']:
    #print(folder + "/" + _file)
    img = load_img(folder + "/" + _file,grayscale=False,target_size=[60,80],interpolation='nearest')  # this is a PIL image
    #img = load_img(folder + "/" + _file)  # this is a PIL image
    #print(img)
    img.thumbnail((image_width, image_height))
    # Convert to Numpy Array
    x = img_to_array(img)  
    x = x.reshape((3, 60, 80))
    #Normalize
    x = (x - 128.0) / 128.0
    dataset[i] = x
    i += 1
    if i % 250 == 0:
        print("%d images to array" % i)
print("All TRAIN images to array!")

j = 0
#print(folder + "/" + df_train['file'])
for _file in df_test['file']:
    #print(folder + "/" + _file)
    img = load_img(folder_test + "/" + _file,grayscale=False,target_size=[60,80],interpolation='nearest')  # this is a PIL image
    #img = load_img(folder + "/" + _file)  # this is a PIL image
    #print(img)
    img.thumbnail((image_width, image_height))
    # Convert to Numpy Array
    x = img_to_array(img)  
    x = x.reshape((3, 60, 80))
    #Normalize
    x = (x - 128.0) / 128.0
    dataset_test[j] = x
    j += 1
    if j % 250 == 0:
        print("%d images to array" % j)
print("All TEST images to array!")

df_train = []
df_test = []

#dataset_test.shape

# This will flatten the entire array of [3,120,160], and also reshape it with right dimention
# https://stackoverflow.com/questions/36967920/numpy-flatten-rgb-image-array
img_flat = dataset.flatten().reshape((dataset.shape[0],3*60*80))   # dataset.shape[0] gives total training size
img_flat_test = dataset_test.flatten().reshape((dataset_test.shape[0],3*60*80))

img_flat.shape
img_flat_test.shape[1]
#Feature Standardization
# mean_px = img_flat.mean().astype(np.float32)
# std_px = img_flat_test.std().astype(np.float32)

# def Standardization(x):
#     return (x-mean_px)/std_px
# DONOT RUN THIS, OUR PROBLEM IS NOT CLASIFICATION PROBLEM
# from keras.utils.np_utils import to_categorical
# df_y = to_categorical(df_y)
# num_classes = df_y.shape[1]
# num_classes
df_y = df_y.values
from keras.models import Sequential
from keras.layers.core import Lambda, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D
model = Sequential()
#model.add(Lambda(Standardization, input_shape =(60,80,3)))
#model.add(Flatten())

#THE INPUT LAYER
model.add(Dense(128,kernel_initializer='normal',input_dim=img_flat.shape[1],  activation='relu'))

#THE HIDDEN LAYER
model.add(Dense(256, kernel_initializer='normal',activation='relu'))
model.add(Dense(256, kernel_initializer='normal',activation='relu'))

#THE OUTPUT LAYER
model.add(Dense(1,kernel_initializer='normal', activation='linear'))



print("Input shape ",model.input_shape)
print("Output shape ",model.output_shape)
# THIS IS FOR CLASSIFICATION PROBLEM
# from keras.optimizers import RMSprop
# model.compile(optimizer=RMSprop(lr=0.01)),
# loss = categorical_crossentropy,
# metrics=['accuracy']

model.compile(loss= 'mean_absolute_error',optimizer = 'adam', metrics = ['mean_absolute_error'])
model.summary()
#DEFINE THE CHECKPOINT
from keras.callbacks import ModelCheckpoint
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]
model.fit(img_flat,df_y,epochs= 10,batch_size = 32, validation_split= 0.2, callbacks= callbacks_list)
wights_file = 'Weights-006--34.30850.hdf5' # choose the best checkpoint 
model.load_weights(wights_file) # load it
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
predictions = model.predict(img_flat_test)
wights_file
predictions.shape
