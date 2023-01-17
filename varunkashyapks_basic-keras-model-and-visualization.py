# Here i used Python 3 environment

# data processing, CSV file 

import pandas as pd

import numpy as np





# Input data files are available in the "../input/" directory.

# The data was stored in the Kaggel website already so i directly check which all data there

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



#  output of the files
# For Visualisation importing the required libraries

# Importing required packages for the graphics

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_json('../input/train.json')

train['inc_angle'] = pd.to_numeric(train['inc_angle'],errors='coerce')

# Checking the columns and head of the data

train.head()
# cheking the data types

train.info()
# Conts of the ice bergs and the ships presesnt in training data

iceberg_ship_count = train['is_iceberg'].value_counts()

iceberg_ship_count1=iceberg_ship_count.plot(kind='bar',colormap='gist_rainbow')

plt.xticks(rotation=25)

iceberg_ship_count1.set_xticklabels( ('Ships', 'Iceberg') )

plt.show()
icebergs = train[train.is_iceberg==1].sample(n=16,random_state=123)



fig = plt.figure(1,figsize=(15,15))

for i in range(16):

    ax = fig.add_subplot(4,4,i+1)

    arr = np.reshape(np.array(icebergs.iloc[i,0]),(75,75))

    ax.imshow(arr,cmap='seismic')

    

plt.show()
icebergs = train[train.is_iceberg==1].sample(n=16,random_state=123)



fig = plt.figure(1,figsize=(15,15))

for i in range(16):

    ax = fig.add_subplot(4,4,i+1)

    arr = np.reshape(np.array(icebergs.iloc[i,0]),(75,75))

    ax.imshow(arr,cmap='magma')

    

plt.show()
ships = train[train.is_iceberg==0].sample(n=9,random_state=456)

fig = plt.figure(1,figsize=(15,15))

for i in range(9):

    ax = fig.add_subplot(3,3,i+1)

    arr = np.reshape(np.array(ships.iloc[i,0]),(75,75))

    ax.imshow(arr,cmap='magma')

    

plt.show()
ships = train[train.is_iceberg==0].sample(n=16,random_state=456)

fig = plt.figure(1,figsize=(15,15))

for i in range(16):

    ax = fig.add_subplot(4,4,i+1)

    arr = np.reshape(np.array(ships.iloc[i,0]),(75,75))

    ax.imshow(arr,cmap='seismic')

    

plt.show()
# Train data

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])

x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])

X_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)

y_train = np.array(train["is_iceberg"])

print("Xtrain:", X_train.shape)
from keras.models import Sequential

from keras.layers import Convolution2D, GlobalAveragePooling2D, Dense, Dropout

model = Sequential()

model.add(Convolution2D(32, 3, activation="relu", input_shape=(75, 75, 2)))

model.add(Convolution2D(64, 3, activation="relu", input_shape=(75, 75, 2)))

model.add(GlobalAveragePooling2D())

model.add(Dropout(0.3))

model.add(Dense(1, activation="sigmoid"))

model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

model.summary()
model.fit(X_train, y_train, validation_split=0.2,epochs = 20)
test = pd.read_json("../input/test.json")
# Test data prediction

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])

x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])

X_test = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)

print("Xtest:", X_test.shape)
prediction = model.predict(X_test, verbose=1)

submit = pd.DataFrame({'id': test["id"], 'is_iceberg': prediction.flatten()})

submit.to_csv("Submitted_first.csv", index=False)
#submit = pd.DataFrame({'id': test["id"], 'is_iceberg': prediction.flatten()})

#submit.to_csv("submitted_2nd.csv", index=False)