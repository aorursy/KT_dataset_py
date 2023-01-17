# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from glob import glob

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.datasets import load_files 
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential


from keras.utils import np_utils

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/flowers/flowers"))

# Any results you write to the current directory are saved as output.
# Make a parent directory `data` and three sub directories `train`, `valid` and 'test'
%rm -rf data # Remove if already present

%mkdir -p data/train/daisy
%mkdir -p data/train/tulip
%mkdir -p data/train/sunflower
%mkdir -p data/train/rose
%mkdir -p data/train/dandelion

%mkdir -p data/valid/daisy
%mkdir -p data/valid/tulip
%mkdir -p data/valid/sunflower
%mkdir -p data/valid/rose
%mkdir -p data/valid/dandelion

%mkdir -p data/test/daisy
%mkdir -p data/test/tulip
%mkdir -p data/test/sunflower
%mkdir -p data/test/rose
%mkdir -p data/test/dandelion


%ls data/train
%ls data/valid
%ls data/test
base_dir = "../input/flowers/flowers"
categories = os.listdir(base_dir)
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from shutil import copyfile

plt.rcParams["figure.figsize"] = (20,3)
def train_valid_test(files):
    """This function splits the files in training, validation and testing sets with 60%, 20%
    and 20% of data in each respectively"""
    train_fles = files[:int(len(files)*0.6)]
    valid_files = files[int(len(files)*0.6):int(len(files)*0.8)]
    test_files = files[int(len(files)*0.8):]
    return train_fles, valid_files, test_files
def copy_files(files, src, dest):
    """This function copy files from src to dest"""
    for file in files:
        copyfile("{}/{}".format(src, file), "{}/{}".format(dest, file))
def plot_images(category, images):
    """This method plots five images from a category"""
    for i in range(len(images)):
        plt.subplot(1,5,i+1)
        plt.title(category)
        image = mpimg.imread("{}/{}/{}".format(base_dir, category, images[i]))
        plt.imshow(image)
    plt.show()
total_images = []
for category in categories:
    images = os.listdir("{}/{}".format(base_dir, category))
    random.shuffle(images)
    filtered_images = [image for image in images if image not in ['flickr.py', 'flickr.pyc', 'run_me.py']]
    
    total_images.append(len(filtered_images))
    
    
    train_images, valid_images, test_images = train_valid_test(filtered_images)
    
    copy_files(train_images, "{}/{}".format(base_dir, category), "./data/train/{}".format(category))
    copy_files(valid_images, "{}/{}".format(base_dir, category), "./data/valid/{}".format(category))
    copy_files(test_images, "{}/{}".format(base_dir, category), "./data/test/{}".format(category))
    plot_images(category, images[:5])
    
        
print("Total images: {}".format(np.sum(total_images)))
for i in range(len(categories)):
    print("{}: {}".format(categories[i], total_images[i]))
y_pos = np.arange(len(categories))
plt.bar(y_pos, total_images, width=0.2,color='b',align='center')
plt.xticks(y_pos, categories)
plt.ylabel("Image count")
plt.title("Image count in different categories")
plt.show()
# define function to load train, valid and test datasets
def load_dataset(path):
    data = load_files(path)
    flower_files = np.array(data['filenames'])
    print(data['target_names'])
    flower_targets = np_utils.to_categorical(np.array(data['target']), 5)
    return flower_files, flower_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('data/train')
valid_files, valid_targets = load_dataset('data/valid')
test_files, test_targets = load_dataset('data/test')

print('There are %d total flower categories.' % len(categories))
print('There are %s total flower images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training flower images.' % len(train_files))
print('There are %d validation flower images.' % len(valid_files))
print('There are %d test flower images.' % len(test_files))

from keras.preprocessing import image                  
from tqdm import tqdm
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)
def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
simple_model = Sequential()
print(train_tensors.shape)

### Define the architecture of the simple model.
simple_model.add(Conv2D(filters=16, kernel_size=2, strides=1, activation='relu', input_shape=(224,224,3)))
simple_model.add(GlobalAveragePooling2D())
simple_model.add(Dense(5, activation='softmax'))
simple_model.summary()
simple_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# Create a `saved_models` directory for saving best model
%mkdir -p saved_models
from keras.callbacks import ModelCheckpoint  

### number of epochs
epochs = 50

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.simple.hdf5', 
                               verbose=1, save_best_only=True)

simple_model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
simple_model.load_weights('saved_models/weights.best.simple.hdf5')
# get index of predicted flower category for each image in test set
flower_predictions = [np.argmax(simple_model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(flower_predictions)==np.argmax(test_targets, axis=1))/len(flower_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
model = Sequential()
print(train_tensors.shape)
### Define architecture.
model.add(Conv2D(filters=16, kernel_size=2, strides=1, activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(filters=32, kernel_size=2, strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(filters=64, kernel_size=2, strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(GlobalAveragePooling2D())
model.add(Dense(5, activation='softmax'))
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint  

### number of epochs
epochs = 50

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
model.load_weights('saved_models/weights.best.from_scratch.hdf5')
# get index of predicted flower category for each image in test set
flower_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(flower_predictions)==np.argmax(test_targets, axis=1))/len(flower_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Model

inception_resnet = InceptionResNetV2(weights="imagenet",include_top=False, input_shape=(224,224,3))
for layer in inception_resnet.layers[:5]:
    layer.trainable = False

output_model = inception_resnet.output
output_model = Flatten()(output_model)
output_model = Dense(200, activation='relu')(output_model)
output_model = Dropout(0.5)(output_model)
output_model = Dense(200, activation='relu')(output_model)
output_model = Dense(5, activation='softmax')(output_model)

model = Model(inputs=inception_resnet.input, outputs=output_model)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint  

### number of epochs
epochs = 50

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.inception_resnetv2.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
### load best weights
model.load_weights('saved_models/weights.best.inception_resnetv2.hdf5')
# get index of predicted flower category for each image in test set 
flower_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(flower_predictions)==np.argmax(test_targets, axis=1))/len(flower_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
for i in range(5):
    predicted = np.argmax(model.predict(np.expand_dims(test_tensors[i], axis=0)))
    actual = np.argmax(test_targets[i])
    print("Predicted: {}, Actual: {}, Name: {}".format(predicted, actual, test_files[i].split("/")[2]))
    image = mpimg.imread(test_files[i])
    plt.imshow(image)
    plt.show()
%rm -rf data
