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
"""

Importing the necessary package

"""



import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split 

from IPython.display import clear_output

from time import sleep

import os
!ls ../input/facial-keypoints-detection/



print("\nExtracting .zip dataset files to working directory ...")

!unzip -u ../input/facial-keypoints-detection/test.zip

!unzip -u ../input/facial-keypoints-detection/training.zip



print("\nCurrent working directory:")

!pwd

print("\nContents of working directory:")

!ls
%%time



train_csv = 'training.csv'

test_csv = 'test.csv'

idlookup_file = '../input/facial-keypoints-detection/IdLookupTable.csv'

train = pd.read_csv(train_csv)

test = pd.read_csv(test_csv)

idlookup_data = pd.read_csv(idlookup_file)
train.head().T
train.info()
print("Total number of images in train : {}".format(len(train)))
train.isnull().sum()
train.fillna(method = 'ffill',inplace = True)
train.isnull().sum()
train.columns
test.isnull().sum()
def load_images(image_data):

    images = []

    for idx, sample in image_data.iterrows():

        image = np.array(sample['Image'].split(' '), dtype=int)

        image = np.reshape(image, (96,96,1))

        images.append(image)

    images = np.array(images)/255.

    return images



def load_keypoints(keypoint_data):

    keypoint_data = keypoint_data.drop('Image',axis = 1)

    keypoint_features = []

    for idx, sample_keypoints in keypoint_data.iterrows():

        keypoint_features.append(sample_keypoints)

    keypoint_features = np.array(keypoint_features, dtype = 'float')

    return keypoint_features



def plot_sample(image, keypoint, axis, title):

    image = image.reshape(96,96)

    axis.imshow(image, cmap='gray')

    axis.scatter(keypoint[0::2], keypoint[1::2], marker='x', s=20)

    plt.title(title)
sample_image_index = 10



clean_train_images = load_images(train)

print("Shape of clean_train_images: {}".format(np.shape(clean_train_images)))

clean_train_keypoints = load_keypoints(train)

print("Shape of clean_train_keypoints: {}".format(np.shape(clean_train_keypoints)))

test_images = load_images(test)

print("Shape of test_images: {}".format(np.shape(test_images)))



train_images = clean_train_images

train_keypoints = clean_train_keypoints

fig, axis = plt.subplots()

plot_sample(clean_train_images[sample_image_index], clean_train_keypoints[sample_image_index], axis, "Sample image & keypoints")
import tensorflow.keras as keras

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import Input,LeakyReLU, Conv2D,Flatten, BatchNormalization, Dense, Dropout, GlobalAveragePooling2D,MaxPool2D

from tensorflow.keras import optimizers

import tensorflow as tf

from keras.utils import np_utils

from keras import applications

from keras.layers import concatenate

import time

from skimage.transform import resize
# #VGG-16 with batch norm and dropout rate = 0.2

# model = Sequential()

# model.add(Conv2D(input_shape=(96,96,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))

# model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))

# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# model.add(Dropout(0.2))

# model.add(BatchNormalization())

# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# model.add(Dropout(0.2))

# model.add(BatchNormalization())

# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# model.add(Dropout(0.4))

# model.add(BatchNormalization())

# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# model.add(BatchNormalization())

# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(GlobalAveragePooling2D())

# model.add(BatchNormalization())



# # Input dimensions: (None, 3, 3, 512)

# model.add(Flatten())

# model.add(Dense(512,activation='relu'))

# model.add(Dropout(0.1))

# model.add(Dense(30))

# model.summary()
# model = Sequential()



# model.add(Conv2D(32, (3,3), padding='same', use_bias=True, input_shape=(96,96,1),activation="relu"))

# model.add(BatchNormalization())

# model.add(Conv2D(32, (3,3), padding='same', use_bias=True,activation="relu"))

# model.add(BatchNormalization())

# model.add(Conv2D(32, (3,3), padding='same', use_bias=True,activation="relu"))

# model.add(BatchNormalization())

# model.add(MaxPool2D(pool_size=(2, 2)))

# # model.add(Dropout(0.1))



# model.add(Conv2D(64, (3,3), padding='same', use_bias=True,activation="relu"))

# model.add(BatchNormalization())

# model.add(Conv2D(64, (3,3), padding='same', use_bias=True,activation="relu"))

# model.add(BatchNormalization())

# model.add(Conv2D(64, (3,3), padding='same', use_bias=True,activation="relu"))

# model.add(BatchNormalization())

# model.add(MaxPool2D(pool_size=(2, 2)))



# model.add(Conv2D(96, (3,3), padding='same', use_bias=True,activation="relu"))

# model.add(BatchNormalization())

# model.add(Conv2D(96, (3,3), padding='same', use_bias=True,activation="relu"))

# model.add(BatchNormalization())

# model.add(Conv2D(96, (3,3), padding='same', use_bias=True,activation="relu"))

# model.add(BatchNormalization())

# model.add(MaxPool2D(pool_size=(2, 2)))

# # model.add(Dropout(0.1))



# model.add(Conv2D(128, (3,3),padding='same', use_bias=True,activation="relu"))

# model.add(BatchNormalization())

# model.add(Conv2D(128, (3,3),padding='same', use_bias=True,activation="relu"))

# model.add(BatchNormalization())

# model.add(Conv2D(128, (3,3),padding='same', use_bias=True,activation="relu"))

# model.add(BatchNormalization())

# model.add(MaxPool2D(pool_size=(2, 2)))



# model.add(Conv2D(256, (3,3),padding='same',use_bias=True,activation="relu"))

# model.add(BatchNormalization())

# model.add(Conv2D(256, (3,3),padding='same',use_bias=True,activation="relu"))

# model.add(BatchNormalization())

# model.add(Conv2D(256, (3,3),padding='same',use_bias=True,activation="relu"))

# model.add(BatchNormalization())

# model.add(MaxPool2D(pool_size=(2, 2)))

# # model.add(Dropout(0.1))



# model.add(Conv2D(512, (3,3), padding='same', use_bias=True,activation="relu"))

# model.add(BatchNormalization())

# model.add(Conv2D(512, (3,3), padding='same', use_bias=True,activation="relu"))

# model.add(BatchNormalization())

# model.add(Conv2D(512, (3,3), padding='same', use_bias=True,activation="relu"))

# model.add(BatchNormalization())





# model.add(Flatten())

# model.add(Dense(512,activation='relu'))

# model.add(Dropout(0.1))

# model.add(Dense(30))

# model.summary()
# BEST MODEL AS NOW 

###########################################

# model = Sequential()



# model.add(Conv2D(32, (3,3), padding='same', use_bias=True, input_shape=(96,96,1)))

# model.add(LeakyReLU(alpha=0.2))

# model.add(BatchNormalization(momentum=0.8))

# model.add(Conv2D(32, (3,3), padding='same', use_bias=True))

# model.add(LeakyReLU(alpha=0.2))

# model.add(BatchNormalization(momentum=0.8))

# model.add(Conv2D(32, (3,3), padding='same', use_bias=True))

# model.add(LeakyReLU(alpha=0.2))

# model.add(BatchNormalization(momentum=0.8))

# model.add(MaxPool2D(pool_size=(2, 2)))

# # model.add(Dropout(0.1))



# model.add(Conv2D(64, (3,3), padding='same', use_bias=True))

# model.add(LeakyReLU(alpha=0.2))

# model.add(BatchNormalization(momentum=0.8))

# model.add(Conv2D(64, (3,3), padding='same', use_bias=True))

# model.add(LeakyReLU(alpha=0.2))

# model.add(BatchNormalization(momentum=0.8))

# model.add(Conv2D(64, (3,3), padding='same', use_bias=True))

# model.add(LeakyReLU(alpha=0.2))

# model.add(BatchNormalization(momentum=0.8))

# model.add(MaxPool2D(pool_size=(2, 2)))



# model.add(Conv2D(96, (3,3), padding='same', use_bias=True))

# model.add(LeakyReLU(alpha=0.2))

# model.add(BatchNormalization(momentum=0.8))

# model.add(Conv2D(96, (3,3), padding='same', use_bias=True))

# model.add(LeakyReLU(alpha=0.2))

# model.add(BatchNormalization(momentum=0.8))

# model.add(Conv2D(96, (3,3), padding='same', use_bias=True))

# model.add(LeakyReLU(alpha=0.2))

# model.add(BatchNormalization(momentum=0.8))

# model.add(MaxPool2D(pool_size=(2, 2)))

# # model.add(Dropout(0.1))



# model.add(Conv2D(128, (3,3),padding='same', use_bias=True))

# model.add(LeakyReLU(alpha=0.2))

# model.add(BatchNormalization(momentum=0.8))

# model.add(Conv2D(128, (3,3),padding='same', use_bias=True))

# model.add(LeakyReLU(alpha=0.2))

# model.add(BatchNormalization(momentum=0.8))

# model.add(MaxPool2D(pool_size=(2, 2)))



# model.add(Conv2D(256, (3,3),padding='same',use_bias=True))

# model.add(LeakyReLU(alpha=0.2))

# model.add(BatchNormalization(momentum=0.8))

# model.add(Conv2D(256, (3,3),padding='same',use_bias=True))

# model.add(LeakyReLU(alpha=0.2))

# model.add(BatchNormalization(momentum=0.8))

# model.add(MaxPool2D(pool_size=(2, 2)))

# # model.add(Dropout(0.1))



# model.add(Conv2D(512, (3,3), padding='same', use_bias=True))

# model.add(LeakyReLU(alpha=0.2))

# model.add(BatchNormalization(momentum=0.8))

# model.add(Conv2D(512, (3,3), padding='same', use_bias=True))

# model.add(LeakyReLU(alpha=0.2))

# model.add(BatchNormalization(momentum=0.8))







# model.add(Flatten())

# model.add(Dense(512))

# model.add(LeakyReLU(alpha=0.2))

# model.add(Dropout(0.1))

# model.add(Dense(30))

# model.summary()
model = Sequential()



model.add(Conv2D(32, (5,5), padding='same', use_bias=True, input_shape=(96,96,1)))

model.add(LeakyReLU(alpha=0.2))

model.add(BatchNormalization(momentum=0.8))

model.add(Conv2D(32, (5,5), padding='same', use_bias=True))

model.add(LeakyReLU(alpha=0.2))

model.add(BatchNormalization(momentum=0.8))

model.add(Conv2D(32, (5,5), padding='same', use_bias=True))

model.add(LeakyReLU(alpha=0.2))

model.add(BatchNormalization(momentum=0.8))

model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Dropout(0.1))



model.add(Conv2D(64, (5,5), padding='same', use_bias=True))

model.add(LeakyReLU(alpha=0.2))

model.add(BatchNormalization(momentum=0.8))

model.add(Conv2D(64, (5,5), padding='same', use_bias=True))

model.add(LeakyReLU(alpha=0.2))

model.add(BatchNormalization(momentum=0.8))

model.add(Conv2D(64, (5,5), padding='same', use_bias=True))

model.add(LeakyReLU(alpha=0.2))

model.add(BatchNormalization(momentum=0.8))

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Conv2D(96, (5,5), padding='same', use_bias=True))

model.add(LeakyReLU(alpha=0.2))

model.add(BatchNormalization(momentum=0.8))

model.add(Conv2D(96, (5,5), padding='same', use_bias=True))

model.add(LeakyReLU(alpha=0.2))

model.add(BatchNormalization(momentum=0.8))

model.add(Conv2D(96, (5,5), padding='same', use_bias=True))

model.add(LeakyReLU(alpha=0.2))

model.add(BatchNormalization(momentum=0.8))

model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Dropout(0.1))



model.add(Conv2D(128, (5,5),padding='same', use_bias=True))

model.add(LeakyReLU(alpha=0.2))

model.add(BatchNormalization(momentum=0.8))

model.add(Conv2D(128, (5,5),padding='same', use_bias=True))

model.add(LeakyReLU(alpha=0.2))

model.add(BatchNormalization(momentum=0.8))

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Conv2D(256, (5,5),padding='same',use_bias=True))

model.add(LeakyReLU(alpha=0.2))

model.add(BatchNormalization(momentum=0.8))

model.add(Conv2D(256, (5,5),padding='same',use_bias=True))

model.add(LeakyReLU(alpha=0.2))

model.add(BatchNormalization(momentum=0.8))

model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Dropout(0.1))



model.add(Conv2D(512, (5,5), padding='same', use_bias=True))

model.add(LeakyReLU(alpha=0.2))

model.add(BatchNormalization(momentum=0.8))

model.add(Conv2D(512, (5,5), padding='same', use_bias=True))

model.add(LeakyReLU(alpha=0.2))

model.add(BatchNormalization(momentum=0.8))





model.add(Flatten())

model.add(Dense(512))

model.add(LeakyReLU(alpha=0.2))

model.add(Dropout(0.1))

model.add(Dense(30))

model.summary()
# (trainX, testX, trainY, testY) = train_test_split(train_images, train_keypoints,

# 	test_size=0.2, random_state=42)

# trainX.shape, testX.shape
# #pyimage source

# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# aug = ImageDataGenerator(

# 		rotation_range=20,

# 		zoom_range=0.15,

# 		width_shift_range=0.2,

# 		height_shift_range=0.2,

# 		shear_range=0.15,

# 		horizontal_flip=True,

# 		fill_mode="nearest")



# aug.fit(train_images)



# X_batch, y_batch = aug.flow(train_images, train_keypoints, batch_size=32)
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from keras.optimizers import Adam



# model = load_model('best_modelV3_1.hdf5') # Uncomment it and start the training from where you left



# Define necessary callbacks

checkpointer = ModelCheckpoint(filepath = 'best_modelV8.hdf5', monitor='val_mae', verbose=1, save_best_only=True, mode='min')

# adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)



# Compile the model

model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mae', 'acc'])



# Train the model

history = model.fit(train_images, train_keypoints, epochs=500, batch_size=256, validation_split=0.1, callbacks=[checkpointer])



# #originally



# from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# from keras.optimizers import Adam



# # model = load_model('best_modelV3_1.hdf5') # Uncomment it and start the training from where you left



# # Define necessary callbacks

# checkpointer = ModelCheckpoint(filepath = 'best_modelV7.hdf5', monitor='val_mae', verbose=1, save_best_only=True, mode='min')

# # adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)



# # Compile the model

# model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mae', 'acc'])



# # Train the model

# history = model.fit(train_images, train_keypoints, epochs=50, batch_size=256, validation_split=0.1, callbacks=[checkpointer])



try:

    plt.plot(history.history['mae'])

    plt.plot(history.history['val_mae'])

    plt.title('Mean Absolute Error vs Epoch')

    plt.ylabel('Mean Absolute Error')

    plt.xlabel('Epochs')

    plt.legend(['train', 'validation'], loc='upper right')

    plt.show()

    # summarize history for accuracy

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.title('Accuracy vs Epoch')

    plt.ylabel('Accuracy')

    plt.xlabel('Epochs')

    plt.legend(['train', 'validation'], loc='upper left')

    plt.show()

    # summarize history for loss

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Loss vs Epoch')

    plt.ylabel('Loss')

    plt.xlabel('Epochs')

    plt.legend(['train', 'validation'], loc='upper left')

    plt.show()

except:

    print("One of the metrics used for plotting graphs is missing! See 'model.compile()'s `metrics` argument.")

from keras.models import load_model

# import h5py

model.save('best_modelV5x5_V1.hdf5')  # creates a HDF5 file 'best_modelV3.hdf5'

# del model  # deletes the existing model



# # returns a compiled model

# # identical to the previous one

%%time



from keras.models import load_model 

model = load_model('best_modelV5x5_V1.hdf5')

test_preds = model.predict(test_images)
fig = plt.figure(figsize=(20,16))

for i in range(10):

    axis = fig.add_subplot(4, 5, i+1, xticks=[], yticks=[])

    plot_sample(test_images[i], test_preds[i], axis, "")

plt.show()
feature_names = list(idlookup_data['FeatureName'])

image_ids = list(idlookup_data['ImageId']-1)

row_ids = list(idlookup_data['RowId'])



feature_list = []

for feature in feature_names:

    feature_list.append(feature_names.index(feature))

    

predictions = []

for x,y in zip(image_ids, feature_list):

    predictions.append(test_preds[x][y])

    

row_ids = pd.Series(row_ids, name = 'RowId')

locations = pd.Series(predictions, name = 'Location')

locations = locations.clip(0.0,96.0)

submission_result = pd.concat([row_ids,locations],axis = 1)

submission_result.to_csv('charlin_version_5X5.csv',index = False)
submission_result
#pyimage source

from tensorflow.keras.preprocessing.image import ImageDataGenerator

aug = ImageDataGenerator(

		rotation_range=20,

		zoom_range=0.15,

		width_shift_range=0.2,

		height_shift_range=0.2,

		shear_range=0.15,

		horizontal_flip=True,

		fill_mode="nearest")
trainX[:,:,:,0].shape

trainX=np.array([trainX[:,:,:,0],trainX[:,:,:,0],trainX[:,:,:,0]])

trainX.shape
testX[:,:,:,0].shape

testX=np.array([testX[:,:,:,0],testX[:,:,:,0],testX[:,:,:,0]])

testX.shape
trainX=np.swapaxes(trainX,0,1)

trainX=np.swapaxes(trainX,1,2)

trainX=np.swapaxes(trainX,2,3)

trainX.shape
testX=np.swapaxes(testX,0,1)

testX=np.swapaxes(testX,1,2)

testX=np.swapaxes(testX,2,3)

testX.shape
# #Resnet Version

# from keras.layers import Input

# img_input = Input(shape=(96,96,1))

# img_conc = tf.keras.layers.Concatenate()([img_input, img_input, img_input])   





# base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (96,96,3))

# base_model.trainable = False

# base_model.summary()



# model=Sequential()

# model.add(Dense(512,activation='relu',input_shape=(2048,)))

# model.add(Dense(256,activation='relu'))

# model.add(Dropout(0.1))

# model.add(Dense(128,activation='relu'))

# model.add(Dropout(0.1))

# model.add(Dense(48,activation='relu'))

# model.add(Dropout(0.1))

# model.add(Dense(30))

# model.summary()

# model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mae', 'acc'])



# final_model = Sequential([base_model, model])

# final_model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mae', 'acc'])

# final_model.summary()
# precomputed_train = base_model.predict(trainX, batch_size=256, verbose=1)

# precomputed_train.shape
# history = model.fit(

#                     x=aug.flow(precomputed_train, trainY,batch_size=256, 

#                                validation_data=[precomputed_val, testY],

#                     steps_per_epoch=len(trainX) // 256,

#                     epochs=10, callbacks=[checkpointer]))
# precomputed_val = base_model.predict(testX,batch_size=256, verbose=1)

# precomputed_val.shape
# from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# from keras.optimizers import Adam



# # model = load_model('best_modelV4.hdf5') # Uncomment it and start the training from where you left



# # Define necessary callbacks

# checkpointer = ModelCheckpoint(filepath = 'best_modelV5.hdf5', monitor='val_mae', verbose=1, save_best_only=True, mode='min')

# # adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)



# # Compile the model

# model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mae', 'acc'])

# # Train the model

# history = model.fit(precomputed_train, trainY, epochs=10, batch_size=256, validation_data=(precomputed_val, testY),callbacks=[checkpointer])

# history = model.fit(

#                     x=aug.flow(precomputed_train, trainY,batch_size=256, 

#                                validation_data=[precomputed_val, testY],

#                     steps_per_epoch=len(trainX) // 256,

#                     epochs=10, callbacks=[checkpointer]))
# try:

#     plt.plot(history.history['mae'])

#     plt.plot(history.history['val_mae'])

#     plt.title('Mean Absolute Error vs Epoch')

#     plt.ylabel('Mean Absolute Error')

#     plt.xlabel('Epochs')

#     plt.legend(['train', 'validation'], loc='upper right')

#     plt.show()

#     # summarize history for accuracy

#     plt.plot(history.history['acc'])

#     plt.plot(history.history['val_acc'])

#     plt.title('Accuracy vs Epoch')

#     plt.ylabel('Accuracy')

#     plt.xlabel('Epochs')

#     plt.legend(['train', 'validation'], loc='upper left')

#     plt.show()

#     # summarize history for loss

#     plt.plot(history.history['loss'])

#     plt.plot(history.history['val_loss'])

#     plt.title('Loss vs Epoch')

#     plt.ylabel('Loss')

#     plt.xlabel('Epochs')

#     plt.legend(['train', 'validation'], loc='upper left')

#     plt.show()

# except:

#     print("One of the metrics used for plotting graphs is missing! See 'model.compile()'s `metrics` argument.")

# from keras.models import load_model

# # import h5py

# model.save('best_modelV5.hdf5')  # creates a HDF5 file 'best_modelV3.hdf5'

# # del model  # deletes the existing model



# # # returns a compiled model

# # # identical to the previous one

# %%time



# from keras.models import load_model 

# model = load_model('best_modelV5.hdf5')

# test_preds = model.predict(test_images)
# fig = plt.figure(figsize=(20,16))

# for i in range(10):

#     axis = fig.add_subplot(4, 5, i+1, xticks=[], yticks=[])

#     plot_sample(test_images[i], test_preds[i], axis, "")

# plt.show()
# feature_names = list(idlookup_data['FeatureName'])

# image_ids = list(idlookup_data['ImageId']-1)

# row_ids = list(idlookup_data['RowId'])



# feature_list = []

# for feature in feature_names:

#     feature_list.append(feature_names.index(feature))

    

# predictions = []

# for x,y in zip(image_ids, feature_list):

#     predictions.append(test_preds[x][y])

    

# row_ids = pd.Series(row_ids, name = 'RowId')

# locations = pd.Series(predictions, name = 'Location')

# locations = locations.clip(0.0,96.0)

# submission_result = pd.concat([row_ids,locations],axis = 1)

# submission_result.to_csv('charlin_version2_6.csv',index = False)
# submission_result