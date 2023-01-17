import matplotlib.pyplot as plt 

import numpy as np

from sklearn.model_selection import train_test_split

from skimage.io import imread

from skimage.transform import pyramid_reduce, resize

import os, glob

import zipfile

from keras.layers import Input, Activation, Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Add, LeakyReLU, UpSampling2D

from keras.models import Model, load_model

from keras.callbacks import ReduceLROnPlateau
# unzip 2d images of lung CT

path = '../input/finding-lungs-in-ct-data/2d_images.zip'

with zipfile.ZipFile(path, 'r') as zip_ref:

    zip_ref.extractall('2d_images')

    

# unzip 2d masks of lung CT

path = '../input/finding-lungs-in-ct-data/2d_masks.zip'

with zipfile.ZipFile(path, 'r') as zip_ref:

    zip_ref.extractall('2d_masks')
# load tiff images

img_list = sorted(glob.glob('2d_images/*.tif'))

mask_list = sorted(glob.glob('2d_masks/*.tif'))



print(len(img_list), len(mask_list))
# create numpy array placeholder for pixels with 1 channel (grey scale)

IMG_SIZE = 256

x_data, y_data = np.empty((2, len(img_list), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

# ^ 1st arg: (# of numpy set(x_data, y_data), length of numpy set, height, width, color channel)
# convert images to numpy arrays

for i, img_path in enumerate(img_list):

    # load image

    img = imread(img_path)

    # resize image with 1 channel

    img = resize(img, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)

    # save to x_data

    x_data[i] = img



# convert mask images to numpy arrays

for i, img_path in enumerate(mask_list):

    # load image

    img = imread(img_path)

    # resize image with 1 channel

    img = resize(img, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)

    # save to x_data

    y_data[i] = img
# scale image arrays

# scaling benefits learning process 

x_data /= 255

y_data /= 255



# display samples of image and mask in gray scale

fig, ax = plt.subplots(1, 2)

ax[0].imshow(x_data[0].squeeze(), cmap='gray')

ax[1].imshow(y_data[0].squeeze(), cmap='gray')

# ^ since imshow takes 2d array, 3d array is squeezed down to 2d by removing single dimensional entry
# split dataset into train and validation datasets with 20% test portion

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2)



print(x_train.shape)

print(x_val.shape)
inputs = Input(shape=(256, 256, 1))



# encoding

net = Conv2D(32, kernel_size=3, activation='relu', padding='same')(inputs)

net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Conv2D(64, kernel_size=3, activation='relu', padding='same')(net)

net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Conv2D(128, kernel_size=3, activation='relu', padding='same')(net)

net = MaxPooling2D(pool_size=2, padding='same')(net)



net = Dense(128, activation='relu')(net)



# decoding

net = UpSampling2D(size=2)(net)

net = Conv2D(128, kernel_size=3, activation='sigmoid', padding='same')(net)

net = UpSampling2D(size=2)(net)

net = Conv2D(64, kernel_size=3, activation='sigmoid', padding='same')(net)

net = UpSampling2D(size=2)(net)



# output with 1 channel for gray scale segmenation

outputs = Conv2D(1, kernel_size=3, activation='sigmoid', padding='same')(net)



model = Model(inputs=inputs, outputs=outputs)



# use binary cross entropy with sigmoid function 

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'mse'])



model.summary()
# reduce learning rate when not improving

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1, min_lr=1e-05)



# train model

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=30, batch_size=32, callbacks=[reduce_lr])
# save model

model.save('image_segmentation.model')
fig, ax = plt.subplots(1, 2, figsize=(10, 7))



# loss

ax[0].set_title('model loss')

ax[0].plot(history.history['loss'], 'b')

ax[0].plot(history.history['val_loss'], 'r')

ax[0].legend(['train', 'test'], loc='upper right')

ax[0].set_ylabel('loss')

ax[0].set_xlabel('epoch')



# accuracy 

ax[1].set_title('model accuracy')

ax[1].plot(history.history['acc'], 'b')

ax[1].plot(history.history['val_acc'], 'r')

ax[1].legend(['train', 'test'], loc='lower right')

ax[1].set_ylabel('accuracy')

ax[1].set_xlabel('epoch')
# predict segmentation

preds = model.predict(x_val)



# show results

fig, ax = plt.subplots(len(x_val), 3, figsize=(10, 100))

for i, pred in enumerate(preds):

    ax[i, 0].imshow(x_val[i].squeeze(), cmap='gray')

    ax[i, 1].imshow(y_val[i].squeeze(), cmap='gray')

    ax[i, 2].imshow(pred.squeeze(), cmap='gray')