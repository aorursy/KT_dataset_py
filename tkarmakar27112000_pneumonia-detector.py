# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/chest_xray/chest_xray'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
base_dir='/kaggle/input/chest_xray/chest_xray'

train_dir=os.path.join(base_dir,'train')

validation_dir=os.path.join(base_dir,'test')

train_normal_dir=os.path.join(train_dir,'NORMAL')

train_pneumonia_dir=os.path.join(train_dir,'PNEUMONIA')

validation_normal_dir=os.path.join(validation_dir,'NORMAL')

validation_pneumonia_dir=os.path.join(validation_dir,'PNEUMONIA')
train_normal_names=os.listdir(train_normal_dir)

train_pneumonia_names=os.listdir(train_pneumonia_dir)

print(train_pneumonia_names[:10])
%matplotlib inline

import tensorflow as tf

import matplotlib.image as mpimg

import matplotlib.pyplot as plt



# Parameters for our graph; we'll output images in a 4x4 configuration

nrows = 4

ncols = 4



pic_index = 0 # Index for iterating over images



# Set up matplotlib fig, and size it to fit 4x4 pics

fig = plt.gcf()

fig.set_size_inches(ncols*4, nrows*4)



pic_index+=8



next_normal_pix = [os.path.join(train_normal_dir, fname) 

                for fname in train_normal_names[ pic_index-8:pic_index] 

               ]



next_pneumonia_pix = [os.path.join(train_pneumonia_dir, fname) 

                for fname in train_pneumonia_names[ pic_index-8:pic_index]

               ]



for i, img_path in enumerate(next_normal_pix+next_pneumonia_pix):

  # Set up subplot; subplot indices start at 1

  sp = plt.subplot(nrows, ncols, i + 1)

  sp.axis('Off') # Don't show axes (or gridlines)



  img = mpimg.imread(img_path)

  plt.imshow(img)



plt.show()

import tensorflow as tf

import numpy as np

from tensorflow import keras

model=tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(1024,activation='relu'),

    tf.keras.layers.Dense(512,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid')

])
model.summary()
from tensorflow.keras.optimizers import RMSprop



model.compile(optimizer=RMSprop(lr=0.001),

             loss='binary_crossentropy',

             metrics=['acc'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

      rescale=1./255,

      rotation_range=60,

      width_shift_range=0.3,

      height_shift_range=0.3,

      shear_range=0.2,

      zoom_range=0.3,

      horizontal_flip=True,

      fill_mode='nearest')



test_datagen=ImageDataGenerator(rescale=1.0/255.)



train_generator=train_datagen.flow_from_directory(train_dir,

                                                 batch_size=20,

                                                 class_mode='binary',

                                                 target_size=(150,150))

validation_generator=train_datagen.flow_from_directory(validation_dir,

                                                 batch_size=20,

                                                 class_mode='binary',

                                                 target_size=(150,150))
history = model.fit_generator(train_generator,

                              validation_data=validation_generator,

                              steps_per_epoch=260,

                              epochs=5,

                              validation_steps=31,

                              verbose=2)
import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'bo', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
import numpy as np

import random

from   tensorflow.keras.preprocessing.image import img_to_array, load_img



# Let's define a new Model that will take an image as input, and will output

# intermediate representations for all layers in the previous model after

# the first.

successive_outputs = [layer.output for layer in model.layers[1:]]



#visualization_model = Model(img_input, successive_outputs)

visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)



# Let's prepare a random input image of a cat or dog from the training set.

cat_img_files = [os.path.join(train_normal_dir, f) for f in train_normal_names]

dog_img_files = [os.path.join(train_pneumonia_dir, f) for f in train_pneumonia_names]



img_path = random.choice(cat_img_files + dog_img_files)

img = load_img(img_path, target_size=(150, 150))  # this is a PIL image



x   = img_to_array(img)                           # Numpy array with shape (150, 150, 3)

x   = x.reshape((1,) + x.shape)                   # Numpy array with shape (1, 150, 150, 3)



# Rescale by 1/255

x /= 255.0



# Let's run our image through our network, thus obtaining all

# intermediate representations for this image.

successive_feature_maps = visualization_model.predict(x)



# These are the names of the layers, so can have them as part of our plot

layer_names = [layer.name for layer in model.layers]



# -----------------------------------------------------------------------

# Now let's display our representations

# -----------------------------------------------------------------------

for layer_name, feature_map in zip(layer_names, successive_feature_maps):

  

  if len(feature_map.shape) == 4:

    

    #-------------------------------------------

    # Just do this for the conv / maxpool layers, not the fully-connected layers

    #-------------------------------------------

    n_features = feature_map.shape[-1]  # number of features in the feature map

    size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)

    

    # We will tile our images in this matrix

    display_grid = np.zeros((size, size * n_features))

    

    #-------------------------------------------------

    # Postprocess the feature to be visually palatable

    #-------------------------------------------------

    for i in range(n_features):

      x  = feature_map[0, :, :, i]

      x -= x.mean()

      x /= x.std ()

      x *=  64

      x += 128

      x  = np.clip(x, 0, 255).astype('uint8')

      display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid



    #-----------------

    # Display the grid

    #-----------------



    scale = 20. / n_features

    plt.figure( figsize=(scale * n_features, scale) )

    plt.title ( layer_name )

    plt.grid  ( False )

    plt.imshow( display_grid, aspect='auto', cmap='viridis' ) 