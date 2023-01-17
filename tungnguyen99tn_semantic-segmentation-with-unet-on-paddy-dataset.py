import tensorflow as tf
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
import numpy as np
import pandas as pd
import os
import rasterio
import rasterio.features
import rasterio.warp
from rasterio.plot import show
from rasterio.mask import mask
import shapely
from fiona.crs import from_epsg
import geopandas as gpd
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
%matplotlib inline
def scale(array):
    arr_min = array.min(axis=(0, 1))
    arr_max = array.max(axis=(0, 1))
    return (array - arr_min) / (arr_max - arr_min)
dataset = rasterio.open('../input/farmboundaries/train/Germany.tif')
polygons = gpd.read_file('../input/farmboundaries/train/field_ger.shp')
print("Dataset's crs ", dataset.crs)
print("Polygons's crs ", polygons.crs)
polygons_mercator = polygons.to_crs({'init': 'epsg:3785'}) 
print("polygons_mercator's crs ", polygons_mercator.crs)
shapes = []
geo = polygons_mercator.values[:, 1]
for i in range(len(polygons_mercator)):
    shapes.append(geo[i])
out = rasterio.mask.raster_geometry_mask(dataset, shapes)
masks = out[0].astype(np.int8)
plt.imshow(masks, cmap='gray')
rgb = dataset.read()
print('shape of rgb: ', rgb.shape)
print('shape of masks: ', masks.shape)
rgb1 = rgb[:, 2560:12800, 13600:20000]
masks1 = masks[2560:12800,  13600:20000]
rgb2 = rgb[:, 0:2560, 0:5120]
masks2 = masks[0:2560,  0:5120]
fig, ax = plt.subplots(1, 2, figsize=(8, 8))
ax[0].imshow(masks1, cmap = 'gray')
ax[0].set_title('Masks 1')
ax[1].imshow(masks2, cmap = 'gray')
ax[1].set_title('Mask 2')
print('shape of rgb1: ', rgb1.shape)
print('shape of masks1: ', masks1.shape)
print('shape of rgb2: ', rgb2.shape)
print('shape of masks2: ', masks2.shape)
rgb1 = np.transpose(rgb1, (1, 2, 0))
rgb1 = scale(rgb1)
rgb2 = np.transpose(rgb2, (1, 2, 0))
rgb2 = scale(rgb2)
rgb1 = resize(rgb1, (10240, 6400, 1), mode = 'constant', preserve_range = True)
rgb2 = resize(rgb2, (2560, 5120, 1), mode = 'constant', preserve_range = True)
masks1 = resize(masks1, (10240, 6400, 1), mode = 'constant', preserve_range = True)
masks2 = resize(masks2, (2560, 5120, 1), mode = 'constant', preserve_range = True)
X = []
y = []
for i in range(0, 10240, 128):
    for j in range(0, 6400, 128):
        X.append(rgb1[i:i+128, j:j+128, :])
        y.append(masks1[i:i+128, j:j+128, :])
for i in range(0, 2560, 128):
    for j in range(0, 5120, 128):
        X.append(rgb2[i:i+128, j:j+128, :])
        y.append(masks2[i:i+128, j:j+128, :])
X = np.asarray(X)
y = np.asarray(y)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=42)
print('X_train: ', X_train.shape)
print('y_train: ', y_train.shape)
print('X_valid: ', X_valid.shape)
print('y_valid: ', y_valid.shape)
def plot_sample(X, y, id):
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(X[id, ..., 0])
    ax[0].set_title('Satellite')
    ax[1].imshow(y[id,..., 0], cmap = 'gray')
    ax[1].set_title('Mask')
plot_sample(X, y, 1997)
import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    # 1st layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 2nd layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x
def get_unet(input_img, n_filters = 16, dropout = 0.05, batchnorm = True):
    """Function to define the UNET Model"""
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
im_width = 128
im_height = 128
input_img = Input((im_height, im_width, 1), name='img')
with tpu_strategy.scope():
    model = get_unet(input_img, n_filters=16, dropout=0.1, batchnorm=True)
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
earlyStopping = EarlyStopping(patience=7, verbose=1)
mcp_save = ModelCheckpoint('farm_model.h5', verbose=1, save_best_only=True, save_weights_only=True)
reduce_lr_loss = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=0)
callbacks = [earlyStopping, mcp_save, reduce_lr_loss]
results = model.fit(X_train, y_train, batch_size=32, epochs=35, callbacks = callbacks, validation_data=(X_valid, y_valid))
plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();
model.evaluate(X_valid, y_valid, verbose=1)
preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
import random
def plot_prediction(X, y, preds, binary_preds, id=None):
    if id is None:
        id = random.randint(0, len(X))

    has_mask = y[id].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[id, ..., 0])
    if has_mask:
        ax[0].contour(y[id].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Satellite')

    ax[1].imshow(y[id].squeeze(), cmap = 'gray')
    ax[1].set_title('Farm')

    ax[2].imshow(preds[id].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[id].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Farm Predicted')
    
    ax[3].imshow(binary_preds[id].squeeze(), vmin=0, vmax=1, cmap = 'gray')
    if has_mask:
        ax[3].contour(y[id].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Farm Predicted binary');
plot_prediction(X_train, y_train, preds_train, preds_train_t)
plot_prediction(X_train, y_train, preds_train, preds_train_t)
plot_prediction(X_train, y_train, preds_train, preds_train_t)