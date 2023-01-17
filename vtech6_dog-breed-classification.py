# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math

import tensorflow as tf

import keras

from keras.utils import to_categorical

from sklearn.preprocessing import OrdinalEncoder

from sklearn.metrics import accuracy_score

from PIL import Image

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path = '/kaggle/input/dog-breed-identification/'

train_path = path+'train/'

test_path = path+'test/'

labels = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')
labels.head()
labels.breed.value_counts()
labels = labels.groupby('breed').head(66).reset_index(drop=True)
labels.breed.value_counts()
datagen = ImageDataGenerator(

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest')



def visualize_df(train_path, path, label_df, new_shape, num_range=range(0, 7), fig_dim=(2, 3)):

    #grayscale_dataset

    

    a=[[0,0], [0, 1], [1, 1], [1,0], [0, 2], [1, 2]]

    fig, axs = plt.subplots(fig_dim[0], fig_dim[1], figsize=(18, 12))

    

    for i in num_range:

    

        image=Image.open(train_path+label_df.id[i]+'.jpg').convert('L')

        image = image.resize((200, 130))

        

        width, height = image.size   # Get dimensions



        left = (width - new_shape)/2

        top = (height - new_shape)/2

        right = (width + new_shape)/2

        bottom = (height + new_shape)/2

        image = image.crop((left, top, right, bottom))

        

        axs[a[i][0], a[i][1]].set_title(label_df.breed[i])

        axs[a[i][0], a[i][1]].axis('off')

        axs[a[i][0], a[i][1]].imshow(image, cmap='gray')



visualize_df(train_path, path, labels, 130, range(0, 6))
def prepare_df(path, label_df, new_shape):

    prepared_labels = []

    prepared_data = []

    

    for i in range(len(label_df)):

        

        image=Image.open(train_path+label_df.id[i]+'.jpg').convert('L')

        image = image.resize((200, 130))

        

        width, height = image.size   # Get dimensions



        left = (width - new_shape)/2

        top = (height - new_shape)/2

        right = (width + new_shape)/2

        bottom = (height + new_shape)/2

        image = image.crop((left, top, right, bottom))

        

        x = img_to_array(image)

        x = x.reshape((1,)+x.shape)

        

        j=0

        

        for batch in datagen.flow(x, batch_size=1):

            j+=1

            if j>4:

                break

            

            prepared_data.append(np.array(batch))

            prepared_labels.append(label_df.breed[i])

        

    return np.array(prepared_data), np.array(prepared_labels)



prepared_dataset, target_labels = prepare_df(train_path, labels, 130)
print(len(prepared_dataset), len(target_labels))
plt.title(target_labels[100])

plt.axis('off')

plt.imshow(prepared_dataset[100].reshape(130, 130))
oe = OrdinalEncoder()

breed = target_labels.reshape(-1, 1)

transformed_labels = oe.fit_transform(breed)
y = to_categorical(transformed_labels)

X = np.array(prepared_dataset)/255
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(len(X_train), len(X_test))

X=0

y=0

prepared_dataset=0
X_train.shape
X_train = X_train.reshape(-1, 130, 130,1)

X_test = X_test.reshape(-1, 130, 130,1)

print(X_train[0].shape, y_train[0].shape)
model = keras.models.Sequential()



model.add(keras.layers.Conv2D(128, (3, 3), strides=(2, 2)))

model.add(keras.layers.Activation('relu'))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))



model.add(keras.layers.BatchNormalization())



model.add(keras.layers.Conv2D(256, (3, 3), strides=(2, 2)))

model.add(keras.layers.Activation('relu'))

model.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))



model.add(keras.layers.BatchNormalization())



model.add(keras.layers.Conv2D(256, (3, 3)))

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Flatten())



model.add(keras.layers.Dense(512))

model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Dense(256))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Dense(128))

model.add(keras.layers.BatchNormalization())



model.add(keras.layers.Dense(120, activation='softmax'))

model.compile(loss='categorical_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])



model.fit(X_train, y_train, epochs=20,validation_data=(X_test, y_test), batch_size=50, shuffle=True)
# from keras.utils import plot_model

# plot_model(model, to_file='model.png')
preds = model.predict_classes(X_test)
accuracy_score(preds, [i.argmax() for i in y_test])