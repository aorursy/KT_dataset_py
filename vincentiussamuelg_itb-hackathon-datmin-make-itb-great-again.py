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
# TRANSFER LEARNING (RESNET50)



from tensorflow.keras.applications import ResNet50

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D



num_classes = 2

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



my_new_model = Sequential()

my_new_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))

my_new_model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

my_new_model.layers[0].trainable = False



# Compile Model

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])



# Fitting Model

from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_size = 224

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)





train_generator = data_generator.flow_from_directory(

        '../input/traingemastik',

        target_size=(image_size, image_size),

        batch_size=24,

        class_mode='categorical')



validation_generator = data_generator.flow_from_directory(

        '../input/testgemastik',

        target_size=(image_size, image_size),

        class_mode='categorical')



my_new_model.fit_generator(

        train_generator,

        steps_per_epoch=3,

        epochs=10,

        validation_data=validation_generator,

        validation_steps=1)
# TRANSFER LEARNING (INCEPTIONV3)



from keras.applications.inception_v3 import InceptionV3

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D



num_classes = 2

weights_path = '../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'



my_new_model = Sequential()

my_new_model.add(InceptionV3(include_top=False, pooling='avg', weights=weights_path))

my_new_model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

my_new_model.layers[0].trainable = False



# Compile Model

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])



# Fitting Model

from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_size = 224

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)





train_generator = data_generator.flow_from_directory(

        '../input/traingemastik',

        target_size=(image_size, image_size),

        batch_size=24,

        class_mode='categorical')



validation_generator = data_generator.flow_from_directory(

        '../input/testgemastik',

        target_size=(image_size, image_size),

        class_mode='categorical')



my_new_model.fit_generator(

        train_generator,

        steps_per_epoch=3,

        epochs=10,

        validation_data=validation_generator,

        validation_steps=1)
# TRANSFER LEARNING WITH AUGMENTATION



from tensorflow.keras.applications import ResNet50

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D



num_classes = 2

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



my_new_model = Sequential()

my_new_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))

my_new_model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

my_new_model.layers[0].trainable = False



# Compile Model

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])



# Fitting Model

from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_size = 224



data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,

                                   horizontal_flip=True,

                                   width_shift_range = 0.2,

                                   height_shift_range = 0.2)



train_generator = data_generator_with_aug.flow_from_directory(

        '../input/traingemastik',

        target_size=(image_size, image_size),

        batch_size=24,

        class_mode='categorical')



data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

validation_generator = data_generator_no_aug.flow_from_directory(

        '../input/testgemastik',

        target_size=(image_size, image_size),

        class_mode='categorical')







history = my_new_model.fit_generator(

        train_generator,

        steps_per_epoch=2,

        epochs=6,

        validation_data=validation_generator,

        validation_steps=1,callbacks=[history])



print(history.history)
import matplotlib.pyplot as plt

loss_train = history.history['loss']

loss_val = history.history['val_loss']

epochs = range(1,13)

plt.plot(epochs, loss_train, 'g', label='Training loss')

plt.plot(epochs, loss_val, 'b', label='validation loss')

plt.title('Training and Validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()