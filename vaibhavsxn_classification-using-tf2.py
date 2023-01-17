# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import zipfile

with zipfile.ZipFile('/kaggle/input/platesv2/plates.zip', 'r') as zip_obj:

   # Extract all the contents of zip file in current directory

   zip_obj.extractall('/kaggle/working/')
print('After zip extraction:')

print(os.listdir("/kaggle/working/"))
data_root = '/kaggle/working/plates/'

print(os.listdir(data_root))
import shutil 

from tqdm import tqdm



train_dir = 'train'

val_dir = 'val'



class_names = ['cleaned', 'dirty']



for dir_name in [train_dir, val_dir]:

    for class_name in class_names:

        os.makedirs(os.path.join(dir_name, class_name), exist_ok=True)



for class_name in class_names:

    source_dir = os.path.join(data_root, 'train', class_name)

    for i, file_name in enumerate(tqdm(os.listdir(source_dir))):

        if i % 6 != 0:

            dest_dir = os.path.join(train_dir, class_name) 

        else:

            dest_dir = os.path.join(val_dir, class_name)

        shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))
!ls train
import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.image import imread
test_path = data_root+'test'

train_path = data_root+'train'
test_path
# os.listdir(test_path)
os.listdir(train_path)
os.listdir(train_path+'/dirty')[0]
one_dirty_plate = train_path+'/dirty'
one_dirty_plate = one_dirty_plate+'/0005.jpg'
one_dirty_plate
one_dirty_plate= imread(one_dirty_plate)
plt.imshow(one_dirty_plate)
train_path
os.listdir(train_path+'/cleaned')[0]
one_clean_plate = train_path+'/cleaned'
one_clean_plate = one_clean_plate+'/0005.jpg'
one_clean_plate= imread(one_clean_plate)
plt.imshow(one_clean_plate)
dirty = '/kaggle/working/train/dirty'

cleaned = '/kaggle/working/train/cleaned'
len(os.listdir(dirty))
len(os.listdir(cleaned))
one_clean_plate.shape, one_dirty_plate.shape
image_shape = one_clean_plate.shape
image_shape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees

                               width_shift_range=0.10, # Shift the pic width by a max of 5%

                               height_shift_range=0.10, # Shift the pic height by a max of 5%

                               rescale=1/255, # Rescale the image by normalzing it.

                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)

                               zoom_range=0.1, # Zoom in by 10% max

                               horizontal_flip=True, # Allo horizontal flipping

                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value

                              )
plt.imshow(one_dirty_plate)
plt.imshow(image_gen.random_transform(one_dirty_plate))
plt.imshow(image_gen.random_transform(one_dirty_plate))
train_path = '/kaggle/working/train/'

test_path = '/kaggle/working/val/'
image_gen.flow_from_directory(train_path);
image_gen.flow_from_directory(test_path)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
#https://stats.stackexchange.com/questions/148139/rules-for-selecting-convolutional-neural-network-hyperparameters

model = Sequential()



model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=image_shape, activation='relu',))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu',))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu',))

model.add(MaxPooling2D(pool_size=(2, 2)))





model.add(Flatten())





model.add(Dense(128))

model.add(Activation('relu'))



# Dropouts help reduce overfitting by randomly turning neurons off during training.

# Here we say randomly turn off 50% of neurons.

model.add(Dropout(0.5))



# Last layer, remember its binary so we use sigmoid

model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
model.summary()
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=2)
help(image_gen.flow_from_directory)
batch_size = 8
train_image_gen = image_gen.flow_from_directory(train_path,

                                               target_size=image_shape[:2],

                                                color_mode='rgb',

                                               batch_size=batch_size,

                                               class_mode='binary')
test_image_gen = image_gen.flow_from_directory(test_path,

                                               target_size=image_shape[:2],

                                               color_mode='rgb',

                                               batch_size=batch_size,

                                               class_mode='binary',shuffle=False)
train_image_gen.class_indices
import warnings

warnings.filterwarnings('ignore')
results = model.fit_generator(train_image_gen,epochs=40,

                              validation_data=test_image_gen,

                             callbacks=[early_stop])
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
model.metrics_names
model.evaluate_generator(test_image_gen)
from tensorflow.keras.preprocessing import image
pred_probabilities = model.predict_generator(test_image_gen)
pred_probabilities
test_image_gen.classes
predictions = pred_probabilities > 0.5
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(test_image_gen.classes,predictions))
confusion_matrix(test_image_gen.classes,predictions)
results = model.fit_generator(train_image_gen,epochs=40,

                              validation_data=test_image_gen)
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
model = Sequential()



model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=image_shape, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(filters=128, kernel_size=(3,3),input_shape=image_shape, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(filters=256, kernel_size=(3,3),input_shape=image_shape, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(filters=512, kernel_size=(3,3),input_shape=image_shape, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



'''

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

'''



model.add(Flatten())





model.add(Dense(64))

model.add(Activation('relu'))



# Dropouts help reduce overfitting by randomly turning neurons off during training.

# Here we say randomly turn off 50% of neurons.

model.add(Dropout(0.5))



# Last layer, remember its binary so we use sigmoid

model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
results = model.fit_generator(train_image_gen,epochs=40,

                              validation_data=test_image_gen)
losses = pd.DataFrame(model.history.history)

losses[['loss','val_loss']].plot()
model.evaluate_generator(test_image_gen)
pred_probabilities = model.predict_generator(test_image_gen)
pred_probabilities
predictions = pred_probabilities > 0.5
confusion_matrix(test_image_gen.classes,predictions)
one_clean_plate = '/kaggle/working/plates/train/cleaned/0000.jpg'
one_clean_plate = image.load_img(one_clean_plate,target_size=image_shape)
one_clean_plate
type(one_clean_plate)
one_clean_plate = image.img_to_array(one_clean_plate)
type(one_clean_plate)
one_clean_plate.shape
one_clean_plate = np.expand_dims(one_clean_plate, axis=0)
one_clean_plate.shape
model.predict(one_clean_plate)
train_image_gen.class_indices
test_image_gen.class_indices