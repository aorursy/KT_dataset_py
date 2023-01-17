# Imports

import pandas as pd

import numpy as np

import cv2 as cv

import matplotlib.pyplot as plt

import seaborn as sns

import os

import gc



from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Conv2D, BatchNormalization, Dropout, MaxPooling2D, Flatten

from keras.layers import PReLU, Activation

from keras.initializers import Constant

from keras.callbacks import ModelCheckpoint



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report
# Lists to store images data

X = []

Y = []



target_size = (75, 75) # The images will be resized to these dimensions



path_to_cat = '../input/cell-images-for-detecting-malaria/cell_images/cell_images/'



for category in os.listdir(path_to_cat):

    path_to_images = os.path.join(path_to_cat, category)

    for im in os.listdir(path_to_images):

        try:

            image_path = os.path.join(path_to_images, im)

            image = cv.imread(image_path)

            image = cv.bilateralFilter(image, 3, 75, 75) # Applying bilateral filter to remove noise

            image = cv.resize(image, target_size) # Resize image        



            X.append(image) 

            Y.append(1) if category == 'Parasitized' else Y.append(0)

        except:

            print(f'ERROR: {category}/{im}')



# Convert lists to np.array and scale data

X = np.array(X).astype('float32')

Y = np.array(Y)



X = X / 255.0



gc.collect()
# The data is splitted on 3 subsets:

# train_x, train_y - to train model

# val_x, val_y - for validtion during training

# test_x, test_y - for final validation

x, test_x, y, test_y = train_test_split(X, Y, test_size = 0.1, stratify = Y, shuffle = True, random_state = 666)

train_x, val_x, train_y, val_y = train_test_split(x, y, test_size = 0.2, stratify = y, shuffle = True, random_state = 666)



# Removing datasets that we don't need anymore to free memory

del X

del Y

del x

del y



print(f'Train data shape: {train_x.shape}, {train_y.shape}')

print(f'Validation data shape: {val_x.shape}, {val_y.shape}')

print(f'Test data shape: {test_x.shape}, {test_y.shape}')



gc.collect()
# Data augmentation

datagen = ImageDataGenerator(rotation_range = 60,

                            shear_range = 10.0,

                            zoom_range = 0.1,

                            fill_mode = 'constant',

                            horizontal_flip = True,

                            vertical_flip = True,)



# Uncomment to see ImageDataGenerator output sample

'''

img = cv.imread('../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/C136P97ThinF_IMG_20151005_140538_cell_96.png')

img = img.reshape(-1, img.shape[0], img.shape[1], 3)

img = img / 255.0



fig = plt.figure(figsize = (18, 10))



for i, flow in enumerate(datagen.flow(img, batch_size = 1)):

    if i > 9:

        break

    fig.add_subplot(2, 5, i+1)

    plt.imshow(np.squeeze(flow)[:, :, ::-1])

    

gc.collect()

'''
def make_model(filters, mode = 'normal'):    

    model = Sequential()

    model.add(Conv2D(filters.pop(0), 5, input_shape = (75, 75, 3), padding = 'valid', kernel_initializer = 'he_normal'))

    model.add(Activation('relu')) if mode == 'normal' else model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(Dropout(0.2))

        

    for i, f in enumerate(filters):        

        model.add(Conv2D(f, 5 if i == 0 else 3, padding = 'valid', kernel_initializer = 'he_normal'))

        model.add(Activation('relu')) if mode == 'normal' else model.add(PReLU(alpha_initializer=Constant(value=0.25)))

        model.add(Dropout(0.2))

        model.add(MaxPooling2D())        

        

    model.add(Flatten())

    model.add(Dense(128, kernel_initializer = 'he_normal'))

    model.add(Activation('relu')) if mode == 'normal' else model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(Dropout(0.5))

    model.add(Dense(64, kernel_initializer = 'he_normal'))

    model.add(Activation('relu')) if mode == 'normal' else model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(Dropout(0.5))

    model.add(Dense(1, activation = 'sigmoid'))

    

    return model
# Model with ReLU

m_relu = make_model([16, 32, 64, 128], mode = 'normal')

m_relu.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

checkpoint = ModelCheckpoint('../working/relu_best.hdf5', verbose = 1, save_best_only = True, save_weights_only = True)

history_relu = m_relu.fit_generator(datagen.flow(train_x, train_y, batch_size = 256),

                             validation_data = [val_x, val_y], 

                             epochs = 60, callbacks = [checkpoint])
# Model with PReLU

m_prelu = make_model([16, 32, 64, 128], mode = 'prelu')

m_prelu.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

checkpoint = ModelCheckpoint('../working/prelu_best.hdf5', verbose = 1, save_best_only = True, save_weights_only = True)

history_prelu = m_prelu.fit_generator(datagen.flow(train_x, train_y, batch_size = 256),

                             validation_data = [val_x, val_y], 

                             epochs = 60, callbacks = [checkpoint])
# Learning curves

fig = plt.figure(figsize = (18, 6))



for i, h in enumerate([m_relu.history, m_prelu.history]):

    fig.add_subplot(1, 2, i+1)

    plt.plot(h.history['accuracy'], label = 'acc')

    plt.plot(h.history['val_accuracy'], label = 'val_acc')

    plt.plot(h.history['loss'], label = 'loss')

    plt.plot(h.history['val_loss'], label = 'val loss')

    plt.legend()

    plt.grid()

    plt.title('Relu') if i == 0 else plt.title('PRelu')

    

plt.show()
# Loading best weights for each model

m_relu.load_weights('../working/relu_best.hdf5')

m_prelu.load_weights('../working/prelu_best.hdf5')
def reports(model):

    fig = plt.figure(figsize = (5, 5))

    preds = model.predict(test_x)

    preds = np.where(preds > 0.5, 1, 0)

    report = classification_report(test_y, preds, output_dict = True)

    confusion = confusion_matrix(test_y, preds)

    sns.heatmap(confusion, fmt = 'd', annot = True, square = True, cbar = False, cmap = 'Blues')    

    print(pd.DataFrame(report))

    plt.show()
# model with ReLU results

reports(m_relu)
# model with PReLU results

reports(m_prelu)