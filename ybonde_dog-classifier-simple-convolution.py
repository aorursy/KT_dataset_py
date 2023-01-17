# importing basic dependencies

import matplotlib.pyplot as plt # for seeing the images

%matplotlib inline

import cv2 # for image processing

import glob # for file handling

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output # to get the files in currect folder

from keras.utils import to_categorical # to convert to one-hot encodings

import tqdm # progress bar

from collections import Counter # for getting breed data



# Importing ML Dependencies

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Dropout, Flatten, Dense

from keras.models import Sequential



'''# Importing ML Dependencies --> Using InceptionV3 as base model

from keras.applications.inception_v3 import InceptionV3 # using this model

from keras.preprocessing import image # preprocessing the images

from keras.models import Model # custom model

from keras.layers import Dense, GlobalAveragePooling2D # layers

from keras import backend as K # backend

from keras.optimizers import SGD # during second compilation, for smoother learning'''



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

files = check_output(["ls", "../input"]).decode("utf8")

print(files)

# Any results you write to the current directory are saved as output.
# loading images path --> train

images_train_path = '../input/train/*.jpg'

images_train_paths = glob.glob(images_train_path)

print(images_train_paths[0])



# laoding images path --> test

images_test_path = '../input/test/*.jpg'

images_test_paths = glob.glob(images_test_path)

print(images_test_paths[0])
# taking the labels for the images

labels = pd.read_csv('../input/labels.csv')

print(labels.head())
# taking the labels and converting to one hot

breeds = sorted(list(set(labels['breed'].values)))

# making a dictionary of breeds which will be used for one-hot encoding

b2id = dict((b,i) for i,b in enumerate(breeds))

# converting labeled breeds to numbers

breed_vector = [b2id[i] for i in labels['breed'].values]

# converting to one-hot encoding

data_y = to_categorical(breed_vector)
print('[*]Total images:', len(images_test_paths) + len(images_train_paths))

print('[*]Total training images:', len(images_train_paths))

print('[*]Total test images:', len(images_test_paths))

print('[*]Total breeds:',len(breeds))

print('[*]data_y.shape:', data_y.shape)
print(data_y[0])
# understanding the distribution of breeds

breed_dict = Counter(labels['breed'].values)

# getting top 5 breeds

breed_numbers = [i for i in breed_dict.values()]

breed_names = [b for b in breed_dict.keys()]
# taking a sample image

img1 = cv2.imread(images_train_paths[120])

plt.imshow(img1)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

plt.imshow(img1)
# Resizing an image to a sqaure

img1 = cv2.resize(img1, (224, 224))

plt.imshow(img1)
# converting to the image to array which will be understood by the model

print(img1.shape)
print('[!]Getting training images:')

total_images_train = np.zeros((len(images_train_paths), 224, 224, 3))

for i in tqdm.tqdm(range(len(images_train_paths))):

    image = cv2.imread(images_train_paths[i]) # reading the image

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # converting to proper colour channel

    image = cv2.resize(image, (224,224)) # resizing to feed into model

    total_images_train[i] = image # adding to the total data
print('[!]Getting testing images:')

total_images_test = np.zeros((len(images_test_paths), 224, 224, 3))

for i in tqdm.tqdm(range(len(images_test_paths))):

    image = cv2.imread(images_test_paths[i]) # reading the image

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # converting to proper colour channel

    image = cv2.resize(image, (224,224)) # resizing to feed into model

    total_images_test[i] = image # adding to the total data
total_images_train = np.array(total_images_train)

# total_images_test = np.array(total_images_test)
print('[*]Traning set shape:', total_images_train.shape)

# print('[*]Testing set shape:', total_images_test.shape)
model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = 2, padding = 'same',

                 activation = 'relu', input_shape = (224, 224, 3)))

model.add(MaxPooling2D(pool_size= 2))

model.add(Conv2D(filters = 32, kernel_size = 2, padding = 'same', activation = 'relu'))

model.add(MaxPooling2D(pool_size= 2))

model.add(Conv2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu'))

model.add(MaxPooling2D(pool_size= 2))

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(1024, activation = 'relu'))

model.add(Dropout(0.4))

model.add(Dense(len(breeds), activation = 'softmax'))



print(model.summary())
# Compiling the model

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(total_images_train, data_y,  epochs = 10, batch_size = 64)