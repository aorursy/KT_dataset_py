# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import zipfile



ba_dir = os.path.join('../input/aksara-jawa/v3/train/ba')

ca_dir = os.path.join('../input/aksara-jawa/v3/train/ca')

da_dir = os.path.join('../input/aksara-jawa/v3/train/da')

dha_dir = os.path.join('../input/aksara-jawa/v3/train/dha')

ga_dir = os.path.join('../input/aksara-jawa/v3/train/ga')

ha_dir = os.path.join('../input/aksara-jawa/v3/train/ha')

ja_dir = os.path.join('../input/aksara-jawa/v3/train/ja')

ka_dir = os.path.join('../input/aksara-jawa/v3/train/ka')

la_dir = os.path.join('../input/aksara-jawa/v3/train/la')

ma_dir = os.path.join('../input/aksara-jawa/v3/train/ma')

na_dir = os.path.join('../input/aksara-jawa/v3/train/na')

nga_dir = os.path.join('../input/aksara-jawa/v3/train/nga')

nya_dir = os.path.join('../input/aksara-jawa/v3/train/nya')

pa_dir = os.path.join('../input/aksara-jawa/v3/train/pa')

ra_dir = os.path.join('../input/aksara-jawa/v3/train/ra')

sa_dir = os.path.join('../input/aksara-jawa/v3/train/sa')

ta_dir = os.path.join('../input/aksara-jawa/v3/train/ta')

tha_dir = os.path.join('../input/aksara-jawa/v3/train/tha')

wa_dir = os.path.join('../input/aksara-jawa/v3/train/wa')

ya_dir = os.path.join('../input/aksara-jawa/v3/train/ya')





print('Total training ba images:', len(os.listdir(ba_dir)))

print('Total training ca images:', len(os.listdir(ca_dir)))

print('Total training da images:', len(os.listdir(da_dir)))

print('Total training dha images:', len(os.listdir(dha_dir)))

print('Total training ga images:', len(os.listdir(ga_dir)))

print('Total training ha images:', len(os.listdir(ha_dir)))

print('Total training ja images:', len(os.listdir(ja_dir)))

print('Total training ka images:', len(os.listdir(ka_dir)))

print('Total training la images:', len(os.listdir(la_dir)))

print('Total training ma images:', len(os.listdir(ma_dir)))

print('Total training na images:', len(os.listdir(na_dir)))

print('Total training nga images:', len(os.listdir(nga_dir)))

print('Total training nya images:', len(os.listdir(nya_dir)))

print('Total training pa images:', len(os.listdir(pa_dir)))

print('Total training ra images:', len(os.listdir(ra_dir)))

print('Total training sa images:', len(os.listdir(sa_dir)))

print('Total training ta images:', len(os.listdir(ta_dir)))

print('Total training tha images:', len(os.listdir(tha_dir)))

print('Total training wa images:', len(os.listdir(wa_dir)))

print('Total training ya images:', len(os.listdir(ya_dir)))







ba_files = os.listdir(ba_dir)

ca_files = os.listdir(ca_dir)

da_files = os.listdir(da_dir)

dha_files = os.listdir(dha_dir)

ga_files = os.listdir(ga_dir)

ha_files = os.listdir(ha_dir)

ja_files = os.listdir(ja_dir)

ka_files = os.listdir(ka_dir)

la_files = os.listdir(la_dir)

ma_files = os.listdir(ma_dir)

na_files = os.listdir(na_dir)

nga_files = os.listdir(nga_dir)

nya_files = os.listdir(nya_dir)

pa_files = os.listdir(pa_dir)

ra_files = os.listdir(ra_dir)

sa_files = os.listdir(sa_dir)

ta_files = os.listdir(ta_dir)

tha_files = os.listdir(tha_dir)

wa_files = os.listdir(wa_dir)

ya_files = os.listdir(ya_dir)





print(ba_files[:5])

print(ca_files[:5])

print(da_files[:5])

print(dha_files[:5])

print(ga_files[:5])

print(ha_files[:5])

print(ja_files[:5])

print(ka_files[:5])

print(la_files[:5])

print(ma_files[:5])

print(na_files[:5])

print(nga_files[:5])

print(nya_files[:5])

print(pa_files[:5])

print(ra_files[:5])

print(sa_files[:5])

print(ta_files[:5])

print(tha_files[:5])

print(wa_files[:5])

print(ya_files[:5])
import tensorflow as tf

import keras_preprocessing

from keras_preprocessing import image

from keras_preprocessing.image import ImageDataGenerator



TRAIN_DIR = '../input/aksara-jawa/v3/train'

training_datagen = ImageDataGenerator(

      rescale = 1./255,

      width_shift_range = 0.2,

      height_shift_range = 0.2,

      shear_range = 0.2,

      zoom_range = 0.2,

      horizontal_flip = True,

      fill_mode = 'nearest'

)





VALIDATION_DIR = '../input/aksara-jawa/v3/test'

validation_datagen = ImageDataGenerator(rescale = 1./255)



train_generator = training_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=(150,150),

    class_mode='categorical',

    batch_size = 16)



validation_generator = validation_datagen.flow_from_directory(

    VALIDATION_DIR,

    target_size = (150,150),

    class_mode = 'categorical',

    batch_size = 16)
local_weights_file = '../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'



pre_trained_model = InceptionV3(

                    input_shape=(150, 150, 3), include_top=False, weights=None)

pre_trained_model.load_weights(local_weights_file)
for layer in pre_trained_model.layers:

    layer.trainable = False
last_layer = pre_trained_model.get_layer('mixed7')

print('last output shape:', last_layer.output_shape)

last_output = last_layer.output
from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras import layers

from tensorflow.keras import Model



# Flatten the output layer to 1 dimension

x = layers.Flatten()(last_output)

# Add a fully connected layer with 1,024 hidden units and ReLU activation

x = layers.Dense(1024, activation='relu')(x)

# Add a dropout rate of 0.2

x = layers.Dropout(0.2)(x)                  

# Add a final sigmoid layer for classification

x = layers.Dense  (20, activation='sigmoid')(x)           



model = Model( pre_trained_model.input, x) 



model.compile(optimizer = RMSprop(lr=0.0001), 

              loss = 'categorical_crossentropy', 

              metrics = ['accuracy'])
# Define a Callback class that stops training once accuracy reaches 90.0%

class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('accuracy')>0.79):

            print("\nReached 79.0% accuracy so cancelling training!")

            self.model.stop_training = True
callback = myCallback()

history = model.fit_generator(

    train_generator,

    steps_per_epoch=20,

    epochs=30,

    validation_data=validation_generator,

    validation_steps=3,

    verbose=1,

    callbacks = callback)
import matplotlib.pyplot as plt

#Evaluasi plot

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc)

plt.plot(epochs, val_acc)

plt.title('Training and Validation accuracy')



plt.figure()



#loss plot

plt.plot(epochs, loss)

plt.plot(epochs, val_loss)

plt.title('Training and Validation Loss')