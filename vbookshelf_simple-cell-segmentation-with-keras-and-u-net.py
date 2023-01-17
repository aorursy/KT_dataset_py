import pandas as pd

import numpy as np

import os



import matplotlib.pyplot as plt

%matplotlib inline



from skimage.io import imread, imshow

from skimage.transform import resize



# Don't Show Warning Messages

import warnings

warnings.filterwarnings('ignore')
IMG_HEIGHT = 128

IMG_WIDTH = 128

IMG_CHANNELS = 1



NUM_TEST_IMAGES = 10

# get a list of files in each folder



img_list = os.listdir('../input/bbbc005_v1_images/BBBC005_v1_images')

mask_list = os.listdir('../input/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth')



# create a dataframe

df_images = pd.DataFrame(img_list, columns=['image_id'])



# filter out the non image file that's called .htaccess

df_images = df_images[df_images['image_id'] != '.htaccess']







# Example file name: SIMCEPImages_A13_C53_F1_s23_w2.TIF





# ======================================================

# Add a column showing how many cells are on each image

# ======================================================



def get_num_cells(x):

    # split on the _

    a = x.split('_')

    # choose the third item

    b = a[2] # e.g. C53

    # choose second item onwards and convert to int

    num_cells = int(b[1:])

    

    return num_cells



# create a new column called 'num_cells'

df_images['num_cells'] = df_images['image_id'].apply(get_num_cells)





# ================================================

# Add a column indicating if an image has a mask.

# ================================================



# Keep in mind images and masks have the same file names.



def check_for_mask(x):

    if x in mask_list:

        return 'yes'

    else:

        return 'no'

    

# create a new column called 'has_mask'

df_images['has_mask'] = df_images['image_id'].apply(check_for_mask)







# ===========================================================

# Add a column showing how much blur was added to each image

# ===========================================================



def get_blur_amt(x):

    # split on the _

    a = x.split('_')

    # choose the third item

    b = a[3] # e.g. F1

    # choose second item onwards and convert to int

    blur_amt = int(b[1:])

    

    return blur_amt



# create a new column called 'blur_amt'

df_images['blur_amt'] = df_images['image_id'].apply(get_blur_amt)
df_images.head(10)
df_masks = df_images[df_images['has_mask'] == 'yes']



# create a new column called mask_id that is just a copy of image_id

df_masks['mask_id'] = df_masks['image_id']



df_masks.shape
df_masks.head()
# create a test set

df_test = df_masks.sample(NUM_TEST_IMAGES, random_state=101)



# Reset the index.

# This is so that we can use loc to access mask id's later.

df_test = df_test.reset_index(drop=True)



# create a list of test images

test_images_list = list(df_test['image_id'])





# Select only rows that are not part of the test set.

# Note the use of ~ to execute 'not in'.

df_masks = df_masks[~df_masks['image_id'].isin(test_images_list)]



print(df_masks.shape)

print(df_test.shape)
# ==================================================== #
sample_image = 'SIMCEPImages_A06_C23_F1_s11_w2.TIF'

path_image = '../input/bbbc005_v1_images/BBBC005_v1_images/' + sample_image



# read the image using skimage

image = imread(path_image)



plt.imshow(image)
print('Shape: ', image.shape)

print('Max pixel value: ', image.max())

print('Min pixel value: ', image.min())
sample_mask = 'SIMCEPImages_A06_C23_F1_s11_w2.TIF'

path_mask = '../input/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth/' + sample_mask



# read the mask using skimage

mask = imread(path_mask)



plt.imshow(mask, cmap='gray')
print('Shape: ', mask.shape)

print('Max pixel value: ', mask.max())

print('Min pixel value: ', mask.min())
# Get lists of images and their masks.

image_id_list = list(df_masks['image_id'])

mask_id_list = list(df_masks['mask_id'])

test_id_list = list(df_test['image_id'])



# Create empty arrays



X_train = np.zeros((len(image_id_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)



Y_train = np.zeros((len(image_id_list), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)



X_test = np.zeros((NUM_TEST_IMAGES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)



# X_train





for i, image_id in enumerate(image_id_list):

    

    path_image = '../input/bbbc005_v1_images/BBBC005_v1_images/' + image_id

    

    # read the image using skimage

    image = imread(path_image)

    

    # resize the image

    image = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    

    # use np.expand dims to add a channel axis so the shape becomes (IMG_HEIGHT, IMG_WIDTH, 1)

    image = np.expand_dims(image, axis=-1)

    

    # insert the image into X_train

    X_train[i] = image

    

X_train.shape
# Y_train





for i, mask_id in enumerate(mask_id_list):

    

    path_mask = '../input/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth/' + mask_id

    

    # read the image using skimage

    mask = imread(path_mask)

    

    # resize the image

    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    

    # use np.expand dims to add a channel axis so the shape becomes (IMG_HEIGHT, IMG_WIDTH, 1)

    mask = np.expand_dims(mask, axis=-1)

    

    # insert the image into Y_Train

    Y_train[i] = mask



Y_train.shape
# X_test



for i, image_id in enumerate(test_id_list):

    

    path_image = '../input/bbbc005_v1_images/BBBC005_v1_images/' + image_id

    

    # read the image using skimage

    image = imread(path_image)

    

    # resize the image

    image = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    

    # use np.expand dims to add a channel axis so the shape becomes (IMG_HEIGHT, IMG_WIDTH, 1)

    image = np.expand_dims(image, axis=-1)

    

    # insert the image into X_test

    X_test[i] = image

    

X_test.shape
from keras.models import Model, load_model

from keras.layers import Input

from keras.layers.core import Dropout, Lambda

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K



import tensorflow as tf
# source: https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277





inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))



s = Lambda(lambda x: x / 255) (inputs)



c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)

c1 = Dropout(0.1) (c1)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)

p1 = MaxPooling2D((2, 2)) (c1)



c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)

c2 = Dropout(0.1) (c2)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)

p2 = MaxPooling2D((2, 2)) (c2)



c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)

c3 = Dropout(0.2) (c3)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)

p3 = MaxPooling2D((2, 2)) (c3)



c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)

c4 = Dropout(0.2) (c4)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)

p4 = MaxPooling2D(pool_size=(2, 2)) (c4)



c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)

c5 = Dropout(0.3) (c5)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)



u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)

u6 = concatenate([u6, c4])

c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)

c6 = Dropout(0.2) (c6)

c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)



u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)

u7 = concatenate([u7, c3])

c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)

c7 = Dropout(0.2) (c7)

c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)



u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)

u8 = concatenate([u8, c2])

c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)

c8 = Dropout(0.1) (c8)

c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)



u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)

u9 = concatenate([u9, c1], axis=3)

c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)

c9 = Dropout(0.1) (c9)

c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)



outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)



model = Model(inputs=[inputs], outputs=[outputs])



model.compile(optimizer='adam', loss='binary_crossentropy')



model.summary()
filepath = "model.h5"



earlystopper = EarlyStopping(patience=5, verbose=1)



checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min')



callbacks_list = [earlystopper, checkpoint]



history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50, 

                    callbacks=callbacks_list)
# Make a prediction



# use the best epoch

model.load_weights('model.h5')



test_preds = model.predict(X_test)
# Threshold the predictions



preds_test_thresh = (test_preds >= 0.5).astype(np.uint8)
# Display a thresholded mask



test_img = preds_test_thresh[5, :, :, 0]



plt.imshow(test_img, cmap='gray')
# set up the canvas for the subplots

plt.figure(figsize=(10,10))

plt.axis('Off')



# Our subplot will contain 3 rows and 3 columns

# plt.subplot(nrows, ncols, plot_number)





# == row 1 ==



# image

plt.subplot(3,3,1)

test_image = X_test[1, :, :, 0]

plt.imshow(test_image)

plt.title('Test Image', fontsize=14)

plt.axis('off')





# true mask

plt.subplot(3,3,2)

mask_id = df_test.loc[1,'mask_id']

path_mask = '../input/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth/' + mask_id

mask = imread(path_mask)

mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

plt.imshow(mask, cmap='gray')

plt.title('True Mask', fontsize=14)

plt.axis('off')



# predicted mask

plt.subplot(3,3,3)

test_mask = preds_test_thresh[1, :, :, 0]

plt.imshow(test_mask, cmap='gray')

plt.title('Pred Mask', fontsize=14)

plt.axis('off')





# == row 2 ==



# image

plt.subplot(3,3,4)

test_image = X_test[2, :, :, 0]

plt.imshow(test_image)

plt.title('Test Image', fontsize=14)

plt.axis('off')





# true mask

plt.subplot(3,3,5)

mask_id = df_test.loc[2,'mask_id']

path_mask = '../input/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth/' + mask_id

mask = imread(path_mask)

mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

plt.imshow(mask, cmap='gray')

plt.title('True Mask', fontsize=14)

plt.axis('off')



# predicted mask

plt.subplot(3,3,6)

test_mask = preds_test_thresh[2, :, :, 0]

plt.imshow(test_mask, cmap='gray')

plt.title('Pred Mask', fontsize=14)

plt.axis('off')



# == row 3 ==



# image

plt.subplot(3,3,7)

test_image = X_test[3, :, :, 0]

plt.imshow(test_image)

plt.title('Test Image', fontsize=14)

plt.axis('off')





# true mask

plt.subplot(3,3,8)

mask_id = df_test.loc[3,'mask_id']

path_mask = '../input/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth/' + mask_id

mask = imread(path_mask)

mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

plt.imshow(mask, cmap='gray')

plt.title('True Mask', fontsize=14)

plt.axis('off')



# predicted mask

plt.subplot(3,3,9)

test_mask = preds_test_thresh[3, :, :, 0]

plt.imshow(test_mask, cmap='gray')

plt.title('Pred Mask', fontsize=14)

plt.axis('off')





plt.tight_layout()

plt.show()