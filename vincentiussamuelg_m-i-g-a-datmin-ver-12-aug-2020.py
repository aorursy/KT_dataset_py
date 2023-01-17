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

from sklearn.metrics import f1_score

import keras



# So that the result is not randomized

seed_value=0



os.environ['PYTHONHASHSEED']=str(seed_value)



import random

random.seed(seed_value)



np.random.seed(seed_value)



import tensorflow as tf 

tf.compat.v1.set_random_seed(seed_value)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

tf.compat.v1.keras.backend.set_session(sess)



num_classes = 3

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



my_new_model = Sequential()

my_new_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))

my_new_model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

my_new_model.layers[0].trainable = False



# Compile Model

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=[tf.keras.metrics.Precision(), 

                        tf.keras.metrics.Recall()])



# Fitting Model

from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_size = 224

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)





train_generator = data_generator.flow_from_directory(

        '../input/traingemastik/train',

        target_size=(image_size, image_size),

        batch_size=24,

        class_mode='categorical')



validation_generator = data_generator.flow_from_directory(

        '../input/testgemastik/test',

        target_size=(image_size, image_size),

        class_mode='categorical')



data_history_Res=my_new_model.fit_generator(

        train_generator,

        steps_per_epoch=3,

        epochs=100,

        validation_data=validation_generator,

        validation_steps=1)

print(data_history_Res.history)
# TRANSFER LEARNING (INCEPTIONV3)



from keras.applications.inception_v3 import InceptionV3

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D 

from keras.callbacks import History 

history = History()



# So that the result is not randomized

seed_value=0



os.environ['PYTHONHASHSEED']=str(seed_value)



import random

random.seed(seed_value)



np.random.seed(seed_value)



import tensorflow as tf 

tf.compat.v1.set_random_seed(seed_value)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

tf.compat.v1.keras.backend.set_session(sess)



num_classes = 3

weights_path = '../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'



my_new_model = Sequential()

my_new_model.add(InceptionV3(include_top=False, pooling='avg', weights=weights_path))

my_new_model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

my_new_model.layers[0].trainable = False



# Compile Model

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=[tf.keras.metrics.Precision(), 

                        tf.keras.metrics.Recall()])



# Fitting Model

from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_size = 224

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)





train_generator = data_generator.flow_from_directory(

        '../input/traingemastik/train',

        target_size=(image_size, image_size),

        batch_size=24,

        class_mode='categorical')



validation_generator = data_generator.flow_from_directory(

        '../input/testgemastik/test',

        target_size=(image_size, image_size),

        class_mode='categorical')



data_history_Inc = my_new_model.fit_generator(

        train_generator,

        steps_per_epoch=3,

        epochs=100,

        validation_data=validation_generator,

        validation_steps=1)

print(data_history_Inc.history)
# TRANSFER LEARNING WITH AUGMENTATION



from tensorflow.keras.applications import ResNet50

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

from keras.callbacks import History 

history = History()



# So that the result is not randomized

seed_value=0



os.environ['PYTHONHASHSEED']=str(seed_value)



import random

random.seed(seed_value)



np.random.seed(seed_value)



import tensorflow as tf 

tf.compat.v1.set_random_seed(seed_value)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

tf.compat.v1.keras.backend.set_session(sess)



num_classes = 3

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



my_new_model = Sequential()

my_new_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))

my_new_model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

my_new_model.layers[0].trainable = False



# Compile Model

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=[tf.keras.metrics.Precision(), 

                        tf.keras.metrics.Recall()])



# Fitting Model

from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_size = 224



data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,

                                   horizontal_flip=True,

                                   width_shift_range = 0.2,

                                   height_shift_range = 0.2)



train_generator = data_generator_with_aug.flow_from_directory(

        '../input/traingemastik/train',

        target_size=(image_size, image_size),

        batch_size=24,

        class_mode='categorical')



data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

validation_generator = data_generator_no_aug.flow_from_directory(

        '../input/testgemastik/test',

        target_size=(image_size, image_size),

        class_mode='categorical')







data_history_Aug = my_new_model.fit_generator(

        train_generator,

        steps_per_epoch=3,

        epochs=100,

        validation_data=validation_generator,

        validation_steps=1,callbacks=[history])



print(data_history_Aug.history)
#Plotting Training and Validation Loss - Augmentation

import matplotlib.pyplot as plt

loss_train_Aug = data_history_Aug.history['loss']

loss_val_Aug = data_history_Aug.history['val_loss']

epochs = range(100)

plt.plot(epochs, loss_train_Aug, 'g', label='Training loss Augmentasi')

plt.plot(epochs, loss_val_Aug, 'b', label='Validation loss Augmentasi')

plt.title('Training and Validation loss - Augmentasi')



plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(loc='upper right')

plt.show()
#Plotting Training and Validation loss - ResNet50

loss_train_Res=data_history_Res.history['loss']

loss_val_Res=data_history_Res.history['val_loss']

plt.plot(epochs,loss_train_Res,'g',label='Training loss ResNet')

plt.plot(epochs,loss_val_Res,'b',label='Validation loss ResNet')

plt.title('Training and Validation loss - ResNet50')



plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(loc='upper right')

plt.show()
#Plotting Training and Validation loss - InceptionV3

loss_train_Inc=data_history_Inc.history['loss']

loss_val_Inc=data_history_Inc.history['val_loss']

plt.plot(epochs,loss_train_Inc,'g',label='Training Loss InceptionV3')

plt.plot(epochs,loss_val_Inc,'b',label='Validation Loss InceptionV3')

plt.title('Training and Validition loss - InceptionV3')



plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(loc='upper right')

plt.show()
#Plotting Val Precision and Recall - ResNet50

val_precision_Res = data_history_Res.history['val_precision_28']

val_recall_Res = data_history_Res.history['val_recall_28']

plt.plot(epochs, val_precision_Res, 'g', label='Validation Precision ResNet50')

plt.plot(epochs, val_recall_Res, 'b', label='Validation Recall ResNet50')

plt.title('Training and Validation loss - ResNet50')

plt.xlabel('Epochs')

plt.ylabel('Score')

plt.legend()

plt.show()
#Plotting Val Precision and Recall - InceptionV3

val_precision_Inc = data_history_Inc.history['val_precision_29']

val_recall_Inc = data_history_Inc.history['val_recall_29']

plt.plot(epochs, val_precision_Inc, 'g', label='Validation Precision InceptionV3')

plt.plot(epochs, val_recall_Inc, 'b', label='Validation Recall InceptionV3')

plt.title('Training and Validation loss - InceptionV3')

plt.xlabel('Epochs')

plt.ylabel('Score')

plt.legend()

plt.show()
#Plotting Val Precision and Recall - Augmentasi

val_precision_Aug = data_history_Aug.history['val_precision_30']

val_recall_Aug = data_history_Aug.history['val_recall_30']

plt.plot(epochs, val_precision_Aug, 'g', label='Validation Precision Augmentasi')

plt.plot(epochs, val_recall_Aug, 'b', label='Validation Recall Augmentasi')

plt.title('Training and Validation loss - Augmentasi')

plt.xlabel('Epochs')

plt.ylabel('Score')

plt.legend()

plt.show()