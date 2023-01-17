# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm, tqdm_notebook

import cv2 as cv



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# read dataset





train_data = pd.read_csv('../input/train.csv')

train_data.head()



training_path = '../input/train/train/'

test_path = '../input/test/test/'



images_train = []

labels_train = []



images = train_data['id'].values

for image_id in tqdm_notebook(images):

    

    image = np.array(cv.imread(training_path + image_id))

    label = train_data[train_data['id'] == image_id]['has_cactus'].values[0]

    

    images_train.append(image)

    labels_train.append(label)

    

    images_train.append(np.flip(image))

    labels_train.append(label)

    

    images_train.append(np.flipud(image))

    labels_train.append(label)

    

    images_train.append(np.fliplr(image))

    labels_train.append(label)

    

    

images_train = np.asarray(images_train)

images_train = images_train.astype('float32')

images_train /= 255.



labels_train = np.asarray(labels_train)



# read test set



test_images_names = []



for filename in os.listdir(test_path):

    test_images_names.append(filename)

    

test_images_names.sort()



images_test = []



for image_id in tqdm_notebook(test_images_names):

    images_test.append(np.array(cv.imread(test_path + image_id)))

    

images_test = np.asarray(images_test)

images_test = images_test.astype('float32')

images_test /= 255
# train/val split





from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(images_train, labels_train, test_size = 0.30, stratify = labels_train)

# augment data

from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

                               width_shift_range=0.1, # Shift the pic width by a max of 10%

                               height_shift_range=0.1, # Shift the pic height by a max of 10%

                               horizontal_flip=True, # Allo horizontal flipping

                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value

                              )

# build the model

from keras.models import Sequential

from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D



model = Sequential()



# let's get 3 convolutional layers



model.add(Conv2D(filters=32, kernel_size=(3,3),

                 input_shape=(32,32,3), activation='relu',))



model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(filters=64, kernel_size=(3,3),

                 input_shape=(32,32,3), activation='relu',))



model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(filters=64, kernel_size=(3,3),

                 input_shape=(32,32,3), activation='relu',))



model.add(MaxPooling2D(pool_size=(2, 2)))





model.add(Flatten())



# fully connected part of the network (158 neurons layer)



model.add(Dense(128))

model.add(Activation('relu'))



# Dropouts help reduce overfitting by randomly turning neurons off during training.

# Here we say randomly turn off 50% of neurons.

model.add(Dropout(0.5))



# Last layer, remember its binary, 0=cat , 1=dog

model.add(Dense(1))

model.add(Activation('sigmoid'))



# we compile the model selecting the loss function, the optimization function and our metric

model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

model.summary()

# define callbacks

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping



weight_name = 'weights_aerial_cactus.h5'



callback = [EarlyStopping(monitor='val_loss', 

                          mode='min', 

                          verbose=1, 

                          patience=10),

            ReduceLROnPlateau(monitor='val_loss', 

                              factor=0.2,

                              patience=5, 

                              min_lr=0.001,

                              verbose=1),

            ModelCheckpoint(filepath = weight_name,

                            save_best_only=True,

                            save_weights_only=True,

                            verbose=1)]
# train the CNN

print('training on: ', len(x_train))

print('validating on: ', len(x_test))



model.fit_generator(

    datagen.flow(x_train, y_train, batch_size=20),

    steps_per_epoch=len(x_train)/20, 

    validation_data=(x_test, y_test),

    epochs=100, 

    callbacks=callback

                   )
# perform prediction

model.load_weights(weight_name)



predictions = model.predict(images_test, verbose = 1)



predictions



# send results

test_df = pd.read_csv('../input/sample_submission.csv')

X_test = []

images_test = test_df['id'].values



for img_id in tqdm_notebook(images_test):

    X_test.append(cv.imread(test_path + img_id))

    

X_test = np.asarray(X_test)

X_test = X_test.astype('float32')

X_test /= 255



y_test_pred = model.predict_proba(X_test)



test_df['has_cactus'] = y_test_pred

test_df.to_csv('aerial-cactus-submission.csv', index = False)
# print report



from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report

import matplotlib.pyplot as plt

import seaborn as sns



x_tr = images_train

y_tr = labels_train



images_train = x_tr

labels_train = y_tr



y_pred_probability = model.predict_proba(x_tr)



y_pred = model.predict_classes(x_tr)

conf_matrix = confusion_matrix(y_tr, y_pred)

fig, ax = plt.subplots(figsize = (10, 10))



sns.heatmap(conf_matrix, annot = True, fmt = 'd', xticklabels = ['0', '1'], yticklabels = ['0', '1'])

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()



print(classification_report(y_tr, y_pred, target_names = ['0','1']))

print("\n\n AUC: {:<0.4f}".format(roc_auc_score(y_tr, y_pred_probability)))