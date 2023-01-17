import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib import colors as mcolors 

import math 

import rasterio

from rasterio.plot import show

from rasterio.mask import mask

import shapely

import geopandas as gpd

import fiona
def scale(array):

    arr_min = array.min(axis=(0, 1))

    arr_max = array.max(axis=(0, 1))

    return (array - arr_min) / (arr_max - arr_min)
dataset = rasterio.open('../input/mi3380-project1-hust/img.tif')

polygons = gpd.read_file('../input/mi3380-project1-hust/shp/paddy_bg.shp')
print("Dataset's crs ", dataset.crs)

print("Polygons's crs ", polygons.crs)
rgb = dataset.read()

rgb.shape
band = []

for i in range (4):

    band.append(dataset.read(i+1))

band = np.asarray(band)
fig, ax = plt.subplots(1, 4, figsize=(13, 13))

ax[0].imshow(band[0])

ax[0].set_title('Band 1')

ax[1].imshow(band[1])

ax[1].set_title('Band 2')

ax[2].imshow(band[2])

ax[2].set_title('Band 3')

ax[3].imshow(band[3])

ax[3].set_title('Band 4 (Near Infrared)')
polygons
poly_background = polygons.iloc[:6, :]

poly_paddy = polygons.iloc[6:, :]
shapes_paddy = []

geo = poly_paddy.values[:, 2]

for i in range(len(poly_paddy)):

    shapes_paddy.append(geo[i])

shapes_background = []

geo = poly_background.values[:, 2]

for i in range(len(poly_background)):

    shapes_background.append(geo[i])  
shapes_paddy
masks_paddy = rasterio.mask.raster_geometry_mask(dataset, shapes_paddy)[0].astype(np.int8)

masks_background = rasterio.mask.raster_geometry_mask(dataset, shapes_background)[0].astype(np.int8)
fig, ax = plt.subplots(1, 2, figsize=(10, 10))

ax[0].imshow(masks_paddy, cmap = 'gray')

ax[0].set_title('Paddy')

ax[1].imshow(masks_background, cmap = 'gray')

ax[1].set_title('Background')
pixels_paddy = []

pixels_background = []

for i in range (len(masks_paddy)):

  for j in range (len(masks_paddy[0])):

    if (masks_paddy[i, j] == 0):

      pixels_paddy.append([i, j])

    if (masks_background[i, j] == 0):

      pixels_background.append([i, j])
samples_paddy = []

samples_background = []



for i in range (len(pixels_paddy)): 

  pixels = np.zeros(4) 

  for j in range (4):

    pixels[j] = dataset.read(j + 1)[pixels_paddy[i][0], pixels_paddy[i][1]]

  samples_paddy.append(pixels)



for i in range (len(pixels_background)): 

  pixels = np.zeros(4) 

  for j in range (4):

    pixels[j] = dataset.read(j + 1)[pixels_background[i][0], pixels_background[i][1]]

  samples_background.append(pixels)



samples_paddy = np.asarray(samples_paddy)

samples_background = np.asarray(samples_background)
print('Shape of samples_paddy: ', samples_paddy.shape)

print('Shape of samples_background: ', samples_background.shape)
df_paddy = pd.DataFrame({ 'Band 1' : samples_paddy[:,0],

                    'Band 2' : samples_paddy[:,1],

                    'Band 3' : samples_paddy[:,2],

                    'Band 4' : samples_paddy[:,3],

                    'Label' : 'paddy'})

df_background = pd.DataFrame({ 'Band 1' : samples_background[:,0],

                    'Band 2' : samples_background[:,1],

                    'Band 3' : samples_background[:,2],

                    'Band 4' : samples_background[:,3],

                    'Label' : 'background'})
print("Paddy Dataframe")

print(df_paddy.head())

print("Shape: ", df_paddy.shape)

print("----------------------------------------------")

print("Background Dataframe")

print(df_background.head())

print("Shape: ", df_background.shape)
def scale(array):

    arr_min = array.min(axis=(0, 1))

    arr_max = array.max(axis=(0, 1))

    return (array - arr_min) / (arr_max - arr_min)
frames = [df_paddy, df_background]

df = pd.concat(frames)
df = pd.get_dummies(df, columns=['Label'])
df.head()
X = df.values[:, 0:4].astype(float)

y = df.values[:, 4:6].astype(float)

X = scale(X)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test=train_test_split(X, y, test_size=0.25, random_state=42)
from keras.models import Sequential 

from keras.layers import Dense 

from keras.optimizers import Adam 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
def create_network():

    model = Sequential()

    model.add(Dense(32, input_shape=(4,), activation='relu'))

    model.add(Dense(16, activation='relu'))

    model.add(Dense(2, activation='softmax'))

        

    return model
earlyStopping = EarlyStopping(patience=10, verbose=0)

mcp_save = ModelCheckpoint('model.h5', verbose=0, save_best_only=True, save_weights_only=True)

reduce_lr_loss = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=0)

callbacks = [earlyStopping, mcp_save, reduce_lr_loss]
model = create_network()

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
results = model.fit(X_train,Y_train, epochs=50, batch_size=32, callbacks = callbacks, validation_data=(X_test, Y_test), verbose = 0)
plt.figure(figsize=(8, 8))

plt.title("Learning curve")

plt.plot(results.history["loss"], label="loss")

plt.plot(results.history["val_loss"], label="val_loss")

plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")

plt.xlabel("Epochs")

plt.ylabel("log_loss")

plt.legend();
model.evaluate(X_test, Y_test, verbose=1)
dataset = rasterio.open('../input/mi3380-project1-hust/img.tif')
rgb = np.transpose(dataset.read(), (1, 2, 0))

rgb.shape
band1 = np.reshape(dataset.read(1), -1)

band2 = np.reshape(dataset.read(2), -1)

band3 = np.reshape(dataset.read(3), -1)

band4 = np.reshape(dataset.read(4), -1)
fully_df = pd.DataFrame({'Band 1': band1,

                    'Band 2' : band2,

                    'Band 3' : band3,

                    'Band 4' : band4})

Observation = scale(fully_df.values)
def model_predict(X):

    result = model.predict(X)

    for i in range (len(X)):

        if (result[i, 1] >= 0.6):

            print(X[i, :], ' is paddy')

        if (result[i, 1] < 0.6 and result[i, 1] > 0.5):

            print(X[i, :], '  may be paddy')

        else:

            print(X[i, :], ' is background')
model_predict(Observation[80:100, :])
def predict(pixel):

    X = np.zeros((1, 4))

    for i in range (4):

        X[0, i] = dataset.read(i+1)[pixel[0], pixel[1]]

    result = model.predict(X)

    if (result[0, 1] >= 0.7):

        print(pixel, ' is paddy')

    if (result[0, 1] < 0.7 and result[0, 1] > 0.5):

        print(pixel, '  may be paddy')

    else:

        print("pixel (",pixel[0],", ", pixel[1],") is background")