dataset_subset = False # Train with only a subset of the data for quick tests

data_subset_size = 2000 # Subset size (for each class)

color = True # Keep the color dimension or else load the data in greyscale

data_generation = True # Perform data augmentation



early_stop_overfitting = True; # Stop the training if the model doesn't improve in order to prevent overfitting

learning_rate_smoothing = True; # Reduces the learning rate of the backpropagation during the fitting if the model isn't improving

#Regular CNNs hiperparameters

batch_size = 16

num_clases = 1

epochs = 100

conv_kernel_size = 3



# CNN fine tuning hiperparameters

default_dropout_rate = 0.2

regularizaion_weight = 0.01

learning_rate_reduction_factor = 0.5



# Data hiperparameters

img_width = 128

img_height = 128



if color:

    img_channels = 3

else:

    img_channels = 1

    
import os # data fetching

import random # training set shuffling

import gc # garbage collector to clean memory

import cv2 # image preprocessing

# Dataset directory check

print(os.listdir("../input/"))

train_dir = '../input/train/train'

test_dir = '../input/test1/test1'



if dataset_subset:

    train_dogs = ['../input/train/train/{}'.format(filename) for filename in os.listdir(train_dir) if 'dog' in filename]

    train_cats = ['../input/train/train/{}'.format(filename) for filename in os.listdir(train_dir) if 'cat' in filename]

    

    # Only a small portion of both classes is used, in favor of quicker 

    train_imgs = train_dogs[:data_subset_size] + train_cats[:data_subset_size]

    

    # Memory freeing tasks

    del train_dogs

    del train_cats

    gc.collect()

    

else:

    train_imgs = ['../input/train/train/{}'.format(filename) for filename in os.listdir(train_dir)]



    

random.shuffle(train_imgs)    

test_imgs = ['../input/test1/test1/{}'.format(filename) for filename in os.listdir(test_dir)]

import numpy as np # linear algebra

from sklearn.model_selection import train_test_split # train-validation splitter
def preprocess_images(img_path_list):

    """

    Loads and preprocesses all the images whose paths included in img_path_list

    Return

        X: array of resized images

        y: array of labels

    """

    X = []

    y = []

    

    for img_path in img_path_list:

        if color:

            x = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

            # This last bit is to have the images coverted from the default BGR from cv2

            # to RGB to correctly visualize the dataset (it has no effect over the training)

        else:

            x = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            

        x = cv2.resize(x, (img_height, img_width))

        X.append(x)

            

        if 'dog' in img_path:

            y.append(1)          

        elif 'cat' in img_path:

            y.append(0)



    return X, y

X, y = preprocess_images(train_imgs)



del train_imgs

gc.collect()



# Validation set splitting

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)



del X

del y

gc.collect()



X_train = np.array(X_train)

X_val = np.array(X_val)

y_train = np.array(y_train)

y_val = np.array(y_val)



train_size = X_train.shape[0]

val_size = X_val.shape[0]



print("Train and validation shapes")

print("X_train: " + str(X_train.shape))

print("X_val: " + str(X_val.shape))

print("y_train: " + str(y_train.shape))

print("y_val: " + str(y_val.shape))

X_test, _ = preprocess_images(test_imgs)



X_test = np.array(X_test)



# Memory freeing tasks

del test_imgs

gc.collect()

    

print("Test dataset shape: ")

print(X_test.shape)

from tensorflow.keras.preprocessing.image import ImageDataGenerator 

# Documentation: https://keras.io/preprocessing/image/

if data_generation:

    

    # Fourth dimension addition in case of its value being onesized

    if img_channels == 1:

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)

        

    data_augmentator = ImageDataGenerator(rescale=1./255, rotation_range=0.2, shear_range=0.1, zoom_range=0.2,

                                          width_shift_range=0.1, height_shift_range=0.1, fill_mode='reflect',horizontal_flip=True)

    data_augmentator.fit(X_train)

    data_generator = data_augmentator.flow(X_train, y_train, batch_size=batch_size)

    

    val_augmentator = ImageDataGenerator(rescale=1./255)

    val_generator = val_augmentator.flow(X_val, y_val, batch_size=batch_size)



    # Dimension restitution

    if img_channels == 1:

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])

        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2])

        
# The good, the bad and the ugly

%matplotlib inline 



from matplotlib import pyplot as plt # data visualization

def plot_data(X, y, num_figures):

    """

    Prints the images stored in X, with their correspondent labels in y.

    num_figures images by row.

    """

    plt.figure(figsize=(30, 20))



    for i in range(num_figures):

        plt.subplot(2, num_figures, i+1)

        if color:

            plt.imshow(X[i])

        else:

            plt.imshow(X[i], cmap='gray')

        if y[i] >= 0.5:

            plt.title("Doge ("+ str(y[i]) + ")", fontsize=30)

        else:

            plt.title("Catto ("+ str(y[i]) + ")", fontsize=30)

            

    plt.tight_layout()

    plt.show()    

# Preprocess training dataset showcase with labels

for i in range(0, 24, 6): 

    plot_data(X_train[i:], y_train[i:], 6)

    
if data_generation:

    

    for X_train_gen, y_train_gen in data_generator:

        

        if img_channels == 1:

            X_train_gen = X_train_gen.reshape(X_train_gen.shape[0], X_train_gen.shape[1], X_train_gen.shape[2])

        

        print("X_train_gen shape: " + str(X_train_gen.shape))

        print("y_train_gen shape: " + str(y_train_gen.shape))

        

        for i in range(0, batch_size-6, 6):

            plot_data(X_train_gen[i:], y_train_gen[i:], 6)

            

        del X_train_gen

        del y_train_gen

        gc.collect()

        

        break



    
# Imports

from tensorflow.keras.models import Sequential # Documentation: https://keras.io/models/sequential/

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # Documentation: https://keras.io/layers/core/, https://keras.io/layers/convolutional/

from tensorflow.keras.layers import Dropout, SpatialDropout2D, BatchNormalization

from tensorflow.keras.optimizers import Adam, RMSprop # Documentation: https://keras.io/optimizers/

from tensorflow.keras.regularizers import l2 # Documentation: https://keras.io/regularizers/

def add_regularization_layer(model, type, rate=default_dropout_rate):

    """

    Adds a regularization layer to the model based on the active control hiperparameters.

    It's open to multipple addition, although you probably want to add only one of them.

    

    'rate' parameter only affects dropout layers.

    """

    if type == "batch_normalization":

        model.add(BatchNormalization())

    if type == "spatial_dropout":

        model.add(SpatialDropout2D(rate)) 

    if type == "dropout":

        model.add(Dropout(rate))
model = Sequential()



model.add(Conv2D(32, kernel_size=(conv_kernel_size, conv_kernel_size), activation='relu', input_shape=(img_width, img_height, img_channels))) # Strides are, by default, (1,1)

add_regularization_layer(model,"batch_normalization")

add_regularization_layer(model,"spatial_dropout", 0.25)

model.add(MaxPooling2D(pool_size=(2,2))) # Strides are, by default, of the same size of the pool size





model.add(Conv2D(64, kernel_size=(conv_kernel_size, conv_kernel_size), activation='relu'))

add_regularization_layer(model,"batch_normalization")

add_regularization_layer(model,"spatial_dropout", 0.25)

model.add(MaxPooling2D(pool_size=(2,2)))





model.add(Conv2D(128, kernel_size=(conv_kernel_size, conv_kernel_size), activation='relu'))

add_regularization_layer(model,"batch_normalization")

add_regularization_layer(model,"spatial_dropout", 0.25)

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(256, kernel_size=(conv_kernel_size, conv_kernel_size), activation='relu'))

add_regularization_layer(model,"batch_normalization")

add_regularization_layer(model,"spatial_dropout", 0.25)

model.add(MaxPooling2D(pool_size=(2,2)))





model.add(Flatten()) 



model.add(Dense(1024, activation='relu', bias_regularizer=l2(regularizaion_weight)))

add_regularization_layer(model,"batch_normalization")

add_regularization_layer(model,"dropout", 0.5)

model.add(Dense(num_clases, activation='sigmoid'))

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Documentation: https://keras.io/callbacks/
# Stops the training in the case of validation score not improving

if early_stop_overfitting:

    early_stop = EarlyStopping(patience=5)

    

    if not learning_rate_smoothing:

        callbacks = [early_stop]



# Reduces the learning rate of the back propagation gradient descend in the case of validation score not improving

if learning_rate_smoothing:

    learning_rate_reduction = ReduceLROnPlateau(monitor="val_acc", patience=2, factor=learning_rate_reduction_factor, min_lr=0.00001, verbose=1)

    

    if not early_stop_overfitting:

        callbacks = [learning_rate_reduction]

    else:

        callbacks = [early_stop, learning_rate_reduction]

model.summary()



model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

if data_generation:

    

    history = model.fit_generator(data_generator, epochs=epochs, validation_data=val_generator, 

                                  steps_per_epoch=train_size//batch_size, validation_steps=val_size//batch_size, 

                                  callbacks=callbacks, verbose=2)



else:

    

    # Fourth dimension addition in case of its value being onesized

    if img_channels == 1:

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)



    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),

                        steps_per_epoch=train_size//batch_size, validation_steps=val_size//batch_size, 

                        callbacks=callbacks, verbose=2)



    # Dimension restitution

    if img_channels == 1:

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])

        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2])

    
# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# Fourth dimension addition in case of its value being onesized

if img_channels == 1:

    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    

predictions = model.predict(X_test, verbose=0)



# Dimension restitution

if img_channels == 1:

    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    
# Preprocess test dataset showcase with predictions

for i in range(0, 60, 6): 

    plot_data(X_test[i:], predictions[i:], 6)

    