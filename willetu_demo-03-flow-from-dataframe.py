# Check the input directories

import os

print(os.listdir('../input'))

print(os.listdir('../input/300_train'))
# Read image names

import pandas as pd

df = pd.read_csv(r"../input/newTrainLabels.csv")

df.head()
# Create image data generator

from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

    rescale=1./255, 

    validation_split = 0.25)



oo = """ All possible features for ImageDataGenerator and their default values

For more details, see: https://keras.io/preprocessing/image/



datagen = ImageDataGenerator(

    featurewise_center = False, 

    samplewise_center = False, 

    featurewise_std_normalization = False, 

    samplewise_std_normalization = False, 

    zca_whitening = False, 

    zca_epsilon = 1e-06, 

    rotation_range = 0, 

    width_shift_range = 0.0, 

    height_shift_range = 0.0, 

    brightness_range = None, 

    shear_range = 0.0, 

    zoom_range = 0.0, 

    channel_shift_range = 0.0, 

    fill_mode = 'nearest', 

    cval = 0.0, 

    horizontal_flip = False, 

    vertical_flip = False, 

    rescale = None, 

    preprocessing_function = None, 

    data_format = None,

    validation_split = 0.0,

    dtype = None)

"""



# Create data generators

train_generator = datagen.flow_from_dataframe(

    dataframe = df, 

    directory = "../input/300_train/300_train",

    has_ext = False,

    x_col = "image", 

    y_col = "level", 

    class_mode = "categorical", 

    target_size = (100, 100), 

    batch_size = 16,

    subset = 'training')



valid_generator = datagen.flow_from_dataframe(

    dataframe = df, 

    directory = "../input/300_train/300_train",

    has_ext = False,

    x_col = "image", 

    y_col = "level", 

    class_mode = "categorical", 

    target_size = (100, 100), 

    batch_size = 16,

    subset = 'validation')



oo = """ All possible settings for flow_from_dataframe

generator = flow_from_dataframe(

    dataframe, 

    directory = None, 

    x_col = 'filename', 

    y_col = 'class', 

    target_size = (256, 256), 

    color_mode = 'rgb', 

    classes = None, 

    class_mode = 'categorical', 

    batch_size = 32, 

    shuffle = True, 

    seed = None, 

    save_to_dir = None, 

    save_prefix = '', 

    save_format = 'png', 

    subset = None, 

    interpolation = 'nearest', 

    drop_duplicates = True)

"""
# Create a basic Sequential model with several Conv2D layers



from keras import Sequential

from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense

from keras import optimizers



model = Sequential()

# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.

# this applies 32 convolution filters of size 3x3 each.

model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (100, 100, 3)))

model.add(Conv2D(32, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation = 'relu'))

model.add(Conv2D(64, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(5, activation = 'softmax'))



# Try a custom metrics, needs to be calculated in backend (Tensorflow)  

from keras import backend

def rmse(y_true, y_pred):

    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))



sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer = sgd,

             loss='categorical_crossentropy', 

              metrics = ["accuracy", "mse", rmse])



oo = """ All possible options for model.compile

For more details, see: https://keras.io/models/model/

model.compile( 

    optimizer, 

    loss = None, 

    metrics = None, 

    loss_weights = None, 

    sample_weight_mode = None, 

    weighted_metrics = None,

    target_tensors = None)

"""



model.summary()
# Calculate how many batches are needed to go through whole train and validation set

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size

STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

N = 10 # Number of epochs



# Train and count seconds

from time import time

t1 = time()

h = model.fit_generator(generator = train_generator,

                    steps_per_epoch = 50, #STEP_SIZE_TRAIN,

                    validation_data = valid_generator,

                    validation_steps = 15, #STEP_SIZE_VALID,

                    epochs = N,

                    verbose = 2)

t2 = time()

elapsed_time = (t2 - t1)



oo = """All settings for model.fit_generator

model.fit_generator(

    generator, 

    steps_per_epoch=None, 

    epochs=1, 

    verbose=1, 

    callbacks=None, 

    validation_data=None, 

    validation_steps=None, 

    validation_freq=1, 

    class_weight=None, 

    max_queue_size=10, 

    workers=1, 

    use_multiprocessing=False, 

    shuffle=True, 

    initial_epoch=0)

"""



# Save the model

model.save('case2.h5')



# Print the total elapsed time and average time per epoch in format (hh:mm:ss)

from time import localtime, strftime

t_total = strftime('%H:%M:%S', localtime(t2 - t1))

t_per_e = strftime('%H:%M:%S', localtime((t2 - t1)/N))

print('Total elapsed time for {:d} epochs: {:s}'.format(N, t_total))

print('Average time per epoch:             {:s}'.format(t_per_e))
# Variable names available in h.history

# loss, val_loss

# acc, val_acc

# mean_squared_error, val_mean_squared_erro

# rmse, val_rmse



%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np



epochs = np.arange(N) + 1.0



f, ax = plt.subplots(2, 2, figsize = (15,10))



def plotter(ax, epochs, h, variable):

    ax.plot(epochs, h.history[variable], label = variable)

    ax.plot(epochs, h.history['val_' + variable], label = 'val_'+variable)

    ax.legend()



plotter(ax[0][0], epochs, h, 'acc')

plotter(ax[0][1], epochs, h, 'loss')

plotter(ax[1][0], epochs, h, 'mean_squared_error')

plotter(ax[1][1], epochs, h, 'rmse')

plt.show()
# Calculate the true and predicted values

# If calculate to whole validation data set, delete/comment line 4 and remove steps from line 5

y_true = valid_generator.classes

y_true = y_true[:16*20] # Take only 20 steps, batch_size = 16

predict = model.predict_generator(valid_generator, steps = 20) # remove steps if calculate to whole dataset

y_pred = np.argmax(predict, axis = 1)
# Calculate and print the metrics results

from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report



cm = confusion_matrix(y_true, y_pred)

print('Confusion matrix:')

print(cm)

print('')



k = cohen_kappa_score(y_true, y_pred, weights = 'quadratic')

print("Quadratic weighted Cohen's kappa = {:.4f}".format(k))

print('')



cr = classification_report(y_true, y_pred)

print('Classification report:')

print(cr)

print('')