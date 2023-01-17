from numpy.random import seed
seed(101)
from tensorflow import set_random_seed
set_random_seed(101)

import pandas as pd
import numpy as np
#import keras

import tensorflow
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import categorical_accuracy

import os
import cv2

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import itertools
import shutil
import matplotlib.pyplot as plt
%matplotlib inline

# To reduce the class imbalance in the train and validation set, the number of 
# rows in class 'healthy' will be reduced to this number:

HEALTHY_SAMPLE_SIZE = 579

# The approx. total number of images we want in each class after doing image augmentation.
# We won't be doing image augmentation on the fly.
NUM_IMAGES_WANTED = 3000 # incl. class 'healthy'

# MobileNet needs input images with shape 224x224x3
IMAGE_SIZE = 224
IMAGE_CHANNELS = 3

IMAGE_PATH = "../input/bee_imgs/bee_imgs/"
# What files are available?

os.listdir("../input")
df = pd.read_csv('../input/bee_data.csv')
df['health'].value_counts()
# Key:
# healthy = healthy
# fvar = few varrao, hive beetles
# var = Varroa, Small Hive Beetles
# ant = ant problems
# robbed = hive being robbed
# queen = missing queen

# Create a new directory
base_dir = 'base_dir'
os.mkdir(base_dir)


#[CREATE FOLDERS INSIDE THE BASE DIRECTORY]

# train_dir
    # 1_healthy
    # 2_fvar
    # 3_var
    # 4_ant
    # 5_robbed
    # 6_queen
 
# val_dir
    # 1_healthy
    # 2_fvar
    # 3_var
    # 4_ant
    # 5_robbed
    # 6_queen

# create a path to 'base_dir' to which we will join the names of the new folders

# train_dir
train_dir = os.path.join(base_dir, 'train_dir')
os.mkdir(train_dir)

# val_dir
val_dir = os.path.join(base_dir, 'val_dir')
os.mkdir(val_dir)


# [CREATE FOLDERS INSIDE THE TRAIN AND VALIDATION FOLDERS]
# Inside each folder we create seperate folders for each class

# create new folders inside train_dir
healthy = os.path.join(train_dir, '1_healthy')
os.mkdir(healthy)
fvar = os.path.join(train_dir, '2_fvar')
os.mkdir(fvar)
var = os.path.join(train_dir, '3_var')
os.mkdir(var)
ant = os.path.join(train_dir, '4_ant')
os.mkdir(ant)
robbed = os.path.join(train_dir, '5_robbed')
os.mkdir(robbed)
queen = os.path.join(train_dir, '6_queen')
os.mkdir(queen)


# create new folders inside val_dir
healthy = os.path.join(val_dir, '1_healthy')
os.mkdir(healthy)
fvar = os.path.join(val_dir, '2_fvar')
os.mkdir(fvar)
var = os.path.join(val_dir, '3_var')
os.mkdir(var)
ant = os.path.join(val_dir, '4_ant')
os.mkdir(ant)
robbed = os.path.join(val_dir, '5_robbed')
os.mkdir(robbed)
queen = os.path.join(val_dir, '6_queen')
os.mkdir(queen)


df_data = pd.read_csv('../input/bee_data.csv')

df_data.head()
# take a random sample of class 'healthy'
df = df_data[df_data['health'] == 'healthy'].sample(HEALTHY_SAMPLE_SIZE, random_state=101)

# remove class 'healthy' from the dataframe
df_data = df_data[df_data['health'] != 'healthy']

# concat df and df_data
df_data = pd.concat([df_data, df], axis=0).reset_index(drop=True)

# shuffle the new dataframe
df_data = shuffle(df_data)

# check the new class distribution
df_data['health'].value_counts()

y = df_data['health']

df_train, df_val = train_test_split(df_data, test_size=0.1, random_state=101, stratify=y)

print(df_train.shape)
print(df_val.shape)
df_train['health'].value_counts()
df_val['health'].value_counts()
# Set the 'file' column as the index in df_data
df_data.set_index('file', inplace=True)

# Get a list of train and val images
train_list = list(df_train['file'])
val_list = list(df_val['file'])

# Key:
# healthy = healthy
# fvar = few varrao, hive beetles
# var = Varroa, Small Hive Beetles
# ant = ant problems
# robbed = hive being robbed
# queen = missing queen



# Transfer the train images.
# Note that we re-name the files during the transfer using a number sequence. This is just a
# safety measure help the generators create a repeatable sequence of images. 
# Otherwise I think they could mix them up and give us confusing results.

length = len(train_list)

for i in range(0,length):
    
    fname = train_list[i]
    image = fname
    description = df_data.loc[image,'health']
    
    # map the class descriptions to folder names
    if description == 'healthy':
        label = '1_healthy'
    if description == 'few varrao, hive beetles':
        label = '2_fvar'
    if description == 'Varroa, Small Hive Beetles':
        label = '3_var'
    if description == 'ant problems':
        label = '4_ant'
    if description == 'hive being robbed':
        label = '5_robbed'
    if description == 'missing queen':
        label = '6_queen'
        
        
    # source path to image
    src = os.path.join('../input/bee_imgs/bee_imgs', fname)
    # chage the file name
    new_name = str(i) + '_' + 'train'+ '.png'
    # destination path to image
    dst = os.path.join(train_dir, label, new_name)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)


# Transfer the val images

length = len(val_list)

for i in range(0,length):
    
    fname = val_list[i]
    image = fname
    description = df_data.loc[image,'health']
    
     # map the class descriptions to folder names
    if description == 'healthy':
        label = '1_healthy'
    if description == 'few varrao, hive beetles':
        label = '2_fvar'
    if description == 'Varroa, Small Hive Beetles':
        label = '3_var'
    if description == 'ant problems':
        label = '4_ant'
    if description == 'hive being robbed':
        label = '5_robbed'
    if description == 'missing queen':
        label = '6_queen'
    
    # source path to image
    src = os.path.join('../input/bee_imgs/bee_imgs', fname)
    # chage the file name
    new_name = str(i) + '_' + 'val' + '.png'
    # destination path to image
    dst = os.path.join(val_dir, label, new_name)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)

        
# check how many train images we have in each folder

print(len(os.listdir('base_dir/train_dir/1_healthy')))
print(len(os.listdir('base_dir/train_dir/2_fvar')))
print(len(os.listdir('base_dir/train_dir/3_var')))
print(len(os.listdir('base_dir/train_dir/4_ant')))
print(len(os.listdir('base_dir/train_dir/5_robbed')))
print(len(os.listdir('base_dir/train_dir/6_queen')))

# check how many val images we have in each folder

print(len(os.listdir('base_dir/val_dir/1_healthy')))
print(len(os.listdir('base_dir/val_dir/2_fvar')))
print(len(os.listdir('base_dir/val_dir/3_var')))
print(len(os.listdir('base_dir/val_dir/4_ant')))
print(len(os.listdir('base_dir/val_dir/5_robbed')))
print(len(os.listdir('base_dir/val_dir/6_queen')))

# note that we are not augmenting class 'healthy'
class_list = ['1_healthy', '2_fvar','3_var','4_ant','5_robbed','6_queen']

for item in class_list:
    
    # We are creating temporary directories here because we delete these directories later.
    
    # create a base dir
    aug_dir = 'aug_dir'
    os.mkdir(aug_dir)
    # create a dir within the base dir to store images of the same class
    img_dir = os.path.join(aug_dir, 'img_dir')
    os.mkdir(img_dir)

    # Choose a class
    img_class = item

    # list all images in that directory
    img_list = os.listdir('base_dir/train_dir/' + img_class)

    # Copy images from the class train dir to the img_dir e.g. class 'healthy'
    for fname in img_list:
            # source path to image
            src = os.path.join('base_dir/train_dir/' + img_class, fname)
            # destination path to image
            dst = os.path.join(img_dir, fname)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)


    # point to a dir containing the images and NOT to the images themselves
    path = aug_dir
    save_path = 'base_dir/train_dir/' + img_class

    # Create a data generator to generate augmented images for each class.
    datagen = ImageDataGenerator(
        #rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.01,
        #horizontal_flip=True,
        #vertical_flip=True,
        brightness_range=(0.9,1.1),
        fill_mode='nearest')

    batch_size = 9

    aug_datagen = datagen.flow_from_directory(path,
                                        save_to_dir=save_path, # this is where the images are saved
                                        save_format='jpg',
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=batch_size)



    # Generate the augmented images and add them to the training folders
    
    # NUM_IMAGES_WANTED = total number of images we want to have in each class
    # We will use image augmentation to create the additional images.
    
    num_files = len(os.listdir(img_dir))
    
    # Just a calculation to get approx. the same amount of images for each class.
    num_batches = int(np.ceil((NUM_IMAGES_WANTED-num_files)/batch_size))

    # Run the generator and create augmented images.
    # Note that these images are automatically stored in a folder. The path
    # to the save folder is specified as a parameter in the generator above.
    for i in range(0,num_batches):

        imgs, labels = next(aug_datagen)
        
    # delete temporary directory with the raw image files
    shutil.rmtree('aug_dir')
# Check how many train images we now have in each folder.
# This is the original images plus the augmented images.

print(len(os.listdir('base_dir/train_dir/1_healthy')))
print(len(os.listdir('base_dir/train_dir/2_fvar')))
print(len(os.listdir('base_dir/train_dir/3_var')))
print(len(os.listdir('base_dir/train_dir/4_ant')))
print(len(os.listdir('base_dir/train_dir/5_robbed')))
print(len(os.listdir('base_dir/train_dir/6_queen')))
# Check how many val images we have in each folder.

print(len(os.listdir('base_dir/val_dir/1_healthy')))
print(len(os.listdir('base_dir/val_dir/2_fvar')))
print(len(os.listdir('base_dir/val_dir/3_var')))
print(len(os.listdir('base_dir/val_dir/4_ant')))
print(len(os.listdir('base_dir/val_dir/5_robbed')))
print(len(os.listdir('base_dir/val_dir/6_queen')))
# plots images with labels within jupyter notebook
# source: https://github.com/smileservices/keras_utils/blob/master/utils.py

def plots(ims, figsize=(12,6), rows=2, interp=False, titles=None): # 12,6
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
        
plots(imgs, titles=None) # titles=labels will display the image labels
# End of Data Preparation
### ===================================================================================== ###
# Start of Model Building
train_path = 'base_dir/train_dir'
valid_path = 'base_dir/val_dir'

num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 10
val_batch_size = 10
image_size = 224

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)


datagen = ImageDataGenerator(preprocessing_function=
                             tensorflow.keras.applications.mobilenet.preprocess_input)

train_gen = datagen.flow_from_directory(train_path,
                                        target_size=(image_size,image_size),
                                        batch_size=train_batch_size,
                                        class_mode='categorical')

val_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(image_size,image_size),
                                        batch_size=val_batch_size,
                                        class_mode='categorical')

# Note: shuffle=False causes the test dataset to not be shuffled.
# Here we will be using the val set as the test dataset because we need to run predict
# in order to generate the confusion matrix.
test_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(image_size,image_size),
                                        batch_size=1,
                                        class_mode='categorical',
                                        shuffle=False)
# Create a copy of a mobilenet model.
# Please ensure your kaggle kernel is set to 'Internet Connected'.

mobile = tensorflow.keras.applications.mobilenet.MobileNet()
mobile.summary()
type(mobile.layers)
# How many layers does MobileNet have?
len(mobile.layers)
# CREATE THE MODEL ARCHITECTURE

# Exclude the last 5 layers of the above model.
# This will include all layers up to and including global_average_pooling2d_1
x = mobile.layers[-6].output

# Create a new dense layer for predictions
# 6 corresponds to the number of classes
#x = Dropout(0.25)(x)
predictions = Dense(6, activation='softmax')(x)

# inputs=mobile.input selects the input layer, outputs=predictions refers to the
# dense layer we created above.

model = Model(inputs=mobile.input, outputs=predictions)
model.summary()
# We need to choose how many layers we actually want to be trained.

# Here we are freezing the weights of all layers except the
# last 23 layers in the new model.
# The last 23 layers of the model will be trained.

for layer in model.layers[:-23]:
    layer.trainable = False
model.compile(Adam(lr=0.001), loss='categorical_crossentropy', 
              metrics=['accuracy'])


filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
                              
callbacks_list = [checkpoint, reduce_lr]

history = model.fit_generator(train_gen, steps_per_epoch=train_steps, 
                            validation_data=val_gen,
                            validation_steps=val_steps,
                            epochs=30, verbose=1,
                           callbacks=callbacks_list)

# Get the metric names so that we can see what the output from evaulate_generator will be.
model.metrics_names
# Here the best epoch will be used.

model.load_weights('model.h5')

val_loss, val_acc = \
model.evaluate_generator(test_gen, 
                        steps=len(df_val))

print('val_loss:', val_loss)
print('val_acc:', val_acc)

# display the loss and accuracy curves

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

# make a prediction
from tensorflow.keras.models import load_model

predictions = model.predict_generator(test_gen, steps=len(df_val), verbose=1)
# Get the index of the class with the highest probability score
y_pred = np.argmax(predictions, axis=1)

# Get the labels of the test images.
y_true = test_gen.classes
# Source: Scikit Learn website
# http://scikit-learn.org/stable/auto_examples/
# model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-
# selection-plot-confusion-matrix-py


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.figure(figsize=(100,100))
    plt.tight_layout()



# argmax returns the index of the max value in a row
cm = confusion_matrix(y_true, y_pred)
test_gen.class_indices
# Key:
# healthy = healthy
# fvar = few varrao, hive beetles
# var = Varroa, Small Hive Beetles
# ant = ant problems
# robbed = hive being robbed
# queen = missing queen

# Define the labels of the class indices. These need to match the 
# order shown above.
cm_plot_labels = ['1_healthy', '2_fvar', '3_var', '4_ant', '5_robbed','6_queen']

plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
len(df_val)
# Get the filenames, labels and associated predictions

# This outputs the sequence in which the generator processed the test images
test_filenames = test_gen.filenames

from sklearn.metrics import classification_report

# Generate a classification report

names = ['healthy', 'few varrao, hive beetles', 'Varroa, Small Hive Beetles', 
               'ant problems', 'hive being robbed','missing queen']

report = classification_report(y_true, y_pred, target_names=names)

print(report)
# End of Model Building
### ===================================================================================== ###
# Convert the Model from Keras to Tensorflow.js
# !pip install tensorflowjs
# create a directory to store the model files
#os.mkdir('tfjs_dir')

# convert to Tensorflow.js
#import tensorflowjs as tfjs

#tensorflowjs.converters.save_keras_model(model, 'tfjs_dir')
# check the the directory containing the model is available
#!ls
# view the files that make up the tensorflow.js model
#os.listdir('tfjs_dir')
# Delete the image data directory we created to prevent a Kaggle error.
# Kaggle allows a max of 500 files to be saved.

shutil.rmtree('base_dir')
