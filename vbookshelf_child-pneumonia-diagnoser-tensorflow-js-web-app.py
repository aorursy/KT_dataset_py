from numpy.random import seed
seed(101)
from tensorflow import set_random_seed
set_random_seed(101)

import pandas as pd
import numpy as np
import keras
from keras import backend as K
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

import os
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt
%matplotlib inline

# Train
# normal
print(len(os.listdir('../input/chest_xray/chest_xray/train/NORMAL')))
# pneumonia
print(len(os.listdir('../input/chest_xray/chest_xray/train/PNEUMONIA')))

# Val
# normal
print(len(os.listdir('../input/chest_xray/chest_xray/val/NORMAL')))
# pneumonia
print(len(os.listdir('../input/chest_xray/chest_xray/val/PNEUMONIA')))
# Test
# normal
print(len(os.listdir('../input/chest_xray/chest_xray/test/NORMAL')))
# pneumonia
print(len(os.listdir('../input/chest_xray/chest_xray/test/PNEUMONIA')))
os.listdir('../input/chest_xray/chest_xray/test')
# create a list of files in each folder
train_normal_list = os.listdir('../input/chest_xray/chest_xray/train/NORMAL')
train_pneu_list = os.listdir('../input/chest_xray/chest_xray/train/PNEUMONIA')
val_normal_list = os.listdir('../input/chest_xray/chest_xray/val/NORMAL')
val_pneu_list = os.listdir('../input/chest_xray/chest_xray/val/PNEUMONIA')
test_normal_list = os.listdir('../input/chest_xray/chest_xray/test/NORMAL')
test_pneu_list = os.listdir('../input/chest_xray/chest_xray/test/PNEUMONIA')
def assign_pneu_type(x):
    x = str(x)
    if 'virus' in x:
        return 'viral'
    if 'bacteria' in x:
        return 'bacterial'
# TRAIN_NORMAL
# create the dataframe
df_train_normal = pd.DataFrame(train_normal_list, columns=['image_id'])
# delete the entry named .DS_Store
df_train_normal = df_train_normal[df_train_normal['image_id'] != '.DS_Store']
# create a new target column
df_train_normal['target'] = 'normal'

# TRAIN_PNEU
# create the dataframe
df_train_pneu = pd.DataFrame(train_pneu_list, columns=['image_id'])
# delete the entry named .DS_Store
df_train_pneu = df_train_pneu[df_train_pneu['image_id'] != '.DS_Store']
# create a target column that's a copy of the image column
df_train_pneu['target'] = df_train_pneu['image_id']
# apply the function to this target column
df_train_pneu['target'] = df_train_pneu['target'].apply(assign_pneu_type)

# VAL_NORMAL
# create the dataframe
df_val_normal = pd.DataFrame(val_normal_list, columns=['image_id'])
# delete the entry named .DS_Store
df_val_normal = df_val_normal[df_val_normal['image_id'] != '.DS_Store']
# create a new target column
df_val_normal['target'] = 'normal'


# VAL_PNEU
# create the dataframe
df_val_pneu = pd.DataFrame(val_pneu_list, columns=['image_id'])
# delete the entry named .DS_Store
df_val_pneu = df_val_pneu[df_val_pneu['image_id'] != '.DS_Store']
# create a target column that's a copy of the image column
df_val_pneu['target'] = df_val_pneu['image_id']
# apply the function to this target column
df_val_pneu['target'] = df_val_pneu['target'].apply(assign_pneu_type)


# TEST_NORMAL
# create the dataframe
df_test_normal = pd.DataFrame(test_normal_list, columns=['image_id'])
# delete the entry named .DS_Store
df_test_normal = df_test_normal[df_test_normal['image_id'] != '.DS_Store']
# create a new target column
df_test_normal['target'] = 'normal'


# TEST_PNEU
# create the dataframe
df_test_pneu = pd.DataFrame(test_pneu_list, columns=['image_id'])
# delete the entry named .DS_Store
df_test_pneu = df_test_pneu[df_test_pneu['image_id'] != '.DS_Store']
# create a target column that's a copy of the image column
df_test_pneu['target'] = df_test_pneu['image_id']
# apply the function to this target column
df_test_pneu['target'] = df_test_pneu['target'].apply(assign_pneu_type)

# Concat the dataframes
df_data = \
pd.concat([df_train_normal,df_train_pneu,df_val_normal,df_val_pneu,df_test_normal,
           df_test_pneu],axis=0).reset_index(drop=True)

# shuffle
df_data = shuffle(df_data)

df_data.shape
# Check the target distribution
df_data['target'].value_counts()
df_data.head()
# Create a new directory
base_dir = 'base_dir'
os.mkdir(base_dir)


#[CREATE FOLDERS INSIDE THE BASE DIRECTORY]

# now we create 3 folders inside 'base_dir':

# train
    # normal
    # bacterial
    # viral

# val
    # normal
    # bacterial
    # viral


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
normal = os.path.join(train_dir, 'normal')
os.mkdir(normal)
bacterial = os.path.join(train_dir, 'bacterial')
os.mkdir(bacterial)
viral = os.path.join(train_dir, 'viral')
os.mkdir(viral)



# create new folders inside val_dir
normal = os.path.join(val_dir, 'normal')
os.mkdir(normal)
bacterial = os.path.join(val_dir, 'bacterial')
os.mkdir(bacterial)
viral = os.path.join(val_dir, 'viral')
os.mkdir(viral)

os.listdir('base_dir/train_dir')
y = df_data['target']

df_train, df_val = train_test_split(df_data, test_size=0.15, random_state=101, stratify=y)

print(df_train.shape)
print(df_val.shape)
# check df_train class distribution
df_train['target'].value_counts()
# check df_val class distribution
df_val['target'].value_counts()
df_data.head()
# Set the image_id as the index in df_data
df_data.set_index('image_id', inplace=True)
# Get a list of images in each of the folders
train_normal_list = os.listdir('../input/chest_xray/chest_xray/train/NORMAL')
train_pneu_list = os.listdir('../input/chest_xray/chest_xray/train/PNEUMONIA')
val_normal_list = os.listdir('../input/chest_xray/chest_xray/val/NORMAL')
val_pneu_list = os.listdir('../input/chest_xray/chest_xray/val/PNEUMONIA')
test_normal_list = os.listdir('../input/chest_xray/chest_xray/test/NORMAL')
test_pneu_list = os.listdir('../input/chest_xray/chest_xray/test/PNEUMONIA')

# Get a list of train and val images
train_list = list(df_train['image_id'])
val_list = list(df_val['image_id'])

# Transfer the train images

for image in train_list:
    
    fname = image
    label = df_data.loc[image,'target']
    
    if fname in train_normal_list:
        # source path to image
        src = os.path.join('../input/chest_xray/chest_xray/train/NORMAL', fname)
        # destination path to image
        dst = os.path.join(train_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

    if fname in train_pneu_list:
        # source path to image
        src = os.path.join('../input/chest_xray/chest_xray/train/PNEUMONIA', fname)
        # destination path to image
        dst = os.path.join(train_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)
    
    if fname in val_normal_list:
        # source path to image
        src = os.path.join('../input/chest_xray/chest_xray/val/NORMAL', fname)
        # destination path to image
        dst = os.path.join(train_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)
    
    if fname in val_pneu_list:
        # source path to image
        src = os.path.join('../input/chest_xray/chest_xray/val/PNEUMONIA', fname)
        # destination path to image
        dst = os.path.join(train_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)
        
    if fname in test_normal_list:
        # source path to image
        src = os.path.join('../input/chest_xray/chest_xray/test/NORMAL', fname)
        # destination path to image
        dst = os.path.join(train_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)
    
    if fname in test_pneu_list:
        # source path to image
        src = os.path.join('../input/chest_xray/chest_xray/test/PNEUMONIA', fname)
        # destination path to image
        dst = os.path.join(train_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)
        
        


# Transfer the val images

for image in val_list:
    
    fname = image
    label = df_data.loc[image,'target']
    
    if fname in train_normal_list:
        # source path to image
        src = os.path.join('../input/chest_xray/chest_xray/train/NORMAL', fname)
        # destination path to image
        dst = os.path.join(val_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

    if fname in train_pneu_list:
        # source path to image
        src = os.path.join('../input/chest_xray/chest_xray/train/PNEUMONIA', fname)
        # destination path to image
        dst = os.path.join(val_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)
    
    if fname in val_normal_list:
        # source path to image
        src = os.path.join('../input/chest_xray/chest_xray/val/NORMAL', fname)
        # destination path to image
        dst = os.path.join(val_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)
    
    if fname in val_pneu_list:
        # source path to image
        src = os.path.join('../input/chest_xray/chest_xray/val/PNEUMONIA', fname)
        # destination path to image
        dst = os.path.join(val_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)
    
    if fname in test_normal_list:
        # source path to image
        src = os.path.join('../input/chest_xray/chest_xray/test/NORMAL', fname)
        # destination path to image
        dst = os.path.join(val_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)
    
    if fname in test_pneu_list:
        # source path to image
        src = os.path.join('../input/chest_xray/chest_xray/test/PNEUMONIA', fname)
        # destination path to image
        dst = os.path.join(val_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)
        



# check how many train images we have in each folder

print(len(os.listdir('base_dir/train_dir/normal')))
print(len(os.listdir('base_dir/train_dir/bacterial')))
print(len(os.listdir('base_dir/train_dir/viral')))
# check how many val images we have in each folder

print(len(os.listdir('base_dir/val_dir/normal')))
print(len(os.listdir('base_dir/val_dir/bacterial')))
print(len(os.listdir('base_dir/val_dir/viral')))

class_list = ['normal','bacterial','viral']

for item in class_list:
    
    # We are creating temporary directories here because we delete these directories later
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

    # Copy images from the class train dir to the img_dir e.g. class 'bacterial'
    for fname in img_list:
            # source path to image
            src = os.path.join('base_dir/train_dir/' + img_class, fname)
            # destination path to image
            dst = os.path.join(img_dir, fname)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)


    # point to a dir containing the images and not to the images themselves
    path = aug_dir
    save_path = 'base_dir/train_dir/' + img_class

    # Create a data generator
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest')

    batch_size = 50

    aug_datagen = datagen.flow_from_directory(path,
                                           save_to_dir=save_path,
                                           save_format='jpg',
                                                    target_size=(224,224),
                                                    batch_size=batch_size)
    
    
    # Generate the augmented images and add them to the training folders
    
    ###########
    
    num_aug_images_wanted = 5000 # total number of images we want to have in each class
    
    ###########
    
    num_files = len(os.listdir(img_dir))
    num_batches = int(np.ceil((num_aug_images_wanted-num_files)/batch_size))

    # run the generator and create augmented images
    for i in range(0,num_batches):

        imgs, labels = next(aug_datagen)
        
    # delete temporary directory with the raw image files
    shutil.rmtree('aug_dir')
# Check how many train images we now have in each folder.
# This is the original images plus the augmented images.

print(len(os.listdir('base_dir/train_dir/normal')))
print(len(os.listdir('base_dir/train_dir/bacterial')))
print(len(os.listdir('base_dir/train_dir/viral')))
# Check how many val images we have in each folder.

print(len(os.listdir('base_dir/val_dir/normal')))
print(len(os.listdir('base_dir/val_dir/bacterial')))
print(len(os.listdir('base_dir/val_dir/viral')))
# plots images with labels within jupyter notebook
# source: https://github.com/smileservices/keras_utils/blob/master/utils.py

def plots(ims, figsize=(12,6), rows=5, interp=False, titles=None): # 12,6
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
train_batches = ImageDataGenerator(
    preprocessing_function= \
    keras.applications.mobilenet.preprocess_input).flow_from_directory(
                                                    train_path,
                                                    target_size=(image_size,image_size),
                                                    batch_size=train_batch_size,
                                                    class_mode='categorical')
valid_batches = ImageDataGenerator(
    preprocessing_function= \
    keras.applications.mobilenet.preprocess_input).flow_from_directory(
                                                    valid_path,
                                                    target_size=(image_size,image_size),
                                                    batch_size=val_batch_size,
                                                    class_mode='categorical')

# Note: shuffle=False causes the test dataset to not be shuffled
test_batches = ImageDataGenerator(
    preprocessing_function= \
    keras.applications.mobilenet.preprocess_input).flow_from_directory(
                                                    valid_path,
                                                    target_size=(image_size,image_size),
                                                    batch_size=val_batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)
# create a copy of a mobilenet model

mobile = keras.applications.mobilenet.MobileNet()
mobile.summary()
type(mobile.layers)
# How many layers does MobileNet have?
len(mobile.layers)
# CREATE THE MODEL ARCHITECTURE

# Exclude the last 5 layers of the above model.
# This will include all layers up to and including global_average_pooling2d_1
x = mobile.layers[-6].output

# Create a new dense layer for predictions
# 3 corresponds to the number of classes
x = Dropout(0.25)(x)
predictions = Dense(3, activation='softmax')(x)

# inputs=mobile.input selects the input layer, outputs=predictions refers to the
# dense layer we created above.

model = Model(inputs=mobile.input, outputs=predictions)
model.summary()
# We need to choose how many layers we actually want to be trained.

# Here we are freezing the weights of all layers except the
# last 40 layers in the new model.
# The last 40 layers of the model will be trained.

for layer in model.layers[:-40]:
    layer.trainable = False
model.compile(Adam(lr=0.001), loss='categorical_crossentropy', 
              metrics=[categorical_accuracy])
# Get the labels that are associated with each index
print(valid_batches.class_indices)
# Add weights to try to make the model more sensitive to a specific class

class_weights={
    0: 1.0, # bacterial
    1: 1.0, # normal
    2: 1.0, # viral
}
filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
                              
callbacks_list = [checkpoint, reduce_lr]

history = model.fit_generator(train_batches, steps_per_epoch=train_steps, 
                              #class_weight=class_weights,
                    validation_data=valid_batches,
                    validation_steps=val_steps,
                    epochs=30, verbose=1,
                   callbacks=callbacks_list)
# get the metric names so we can use evaulate_generator
model.metrics_names
# Here the the last epoch will be used.

val_loss, val_cat_acc = \
model.evaluate_generator(test_batches, 
                        steps=val_steps)

print('val_loss:', val_loss)
print('val_cat_acc:', val_cat_acc)
# Here the best epoch will be used.

model.load_weights('model.h5')

val_loss, val_cat_acc = \
model.evaluate_generator(test_batches, 
                        steps=val_steps)

print('val_loss:', val_loss)
print('val_cat_acc:', val_cat_acc)
# display the loss and accuracy curves

import matplotlib.pyplot as plt

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.plot(epochs, acc, 'bo', label='Training cat acc')
plt.plot(epochs, val_acc, 'b', label='Validation cat acc')
plt.title('Training and validation cat accuracy')
plt.legend()
plt.figure()
# Get the labels of the test images.

test_labels = test_batches.classes
# We need these to plot the confusion matrix.
test_labels
# Print the label associated with each class
test_batches.class_indices
# make a prediction
predictions = model.predict_generator(test_batches, steps=val_steps, verbose=1)
predictions.shape
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
    plt.tight_layout()
test_labels.shape
# argmax returns the index of the max value in a row
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
test_batches.class_indices
# Define the labels of the class indices. These need to match the 
# order shown above.
cm_plot_labels = ['bacterial', 'normal', 'viral']

plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
# Get the filenames, labels and associated predictions

# This outputs the sequence in which the generator processed the test images
test_filenames = test_batches.filenames

# Get the true labels
y_true = test_batches.classes

# Get the predicted labels
y_pred = predictions.argmax(axis=1)
from sklearn.metrics import classification_report

# Generate a classification report


report = classification_report(y_true, y_pred, target_names=cm_plot_labels)

print(report)
# Get the filenames, labels and associated predictions

test_filenames = test_batches.filenames
test_labels = test_batches.classes
preds = predictions.argmax(axis=1)


# check the lengths of these lists
print(len(test_filenames))
print(len(test_labels))
print(len(preds))
# Put the above into a dataframe
pred_dict = {'filenames': test_filenames, 'labels': test_labels, 'predictions': preds}
df_preds = pd.DataFrame(pred_dict)

df_preds.head()
# get the indices for the labels
test_batches.class_indices
# filter out rows where the label was bacterial (0) and the model predicted viral (2)
df_1 = df_preds[(df_preds['labels'] == 0) & (df_preds['predictions'] == 2)]

# reset the index
df_1.reset_index(inplace=True, drop=True)

df_1.head()
img_0 = val_dir + '/' + df_1.loc[0, 'filenames']
img_1 = val_dir + '/' + df_1.loc[1, 'filenames']
img_2 = val_dir + '/' + df_1.loc[2, 'filenames']
img_3 = val_dir + '/' + df_1.loc[3, 'filenames']
# These are 4 bacterial pneumonia images that the model mis-classified as viral pneumonia.

plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plt.imshow(plt.imread(img_0), cmap='gray')
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(plt.imread(img_1), cmap='gray')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(plt.imread(img_2), cmap='gray')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(plt.imread(img_3), cmap='gray')
plt.axis('off')
# Now let's print some true viral pneumonia images

df_2 = df_preds[(df_preds['labels'] == 2) & (df_preds['predictions'] == 2)]

# reset the index
df_2.reset_index(inplace=True, drop=True)

df_2.head()
img_0 = val_dir + '/' + df_2.loc[0, 'filenames']
img_1 = val_dir + '/' + df_2.loc[1, 'filenames']
img_2 = val_dir + '/' + df_2.loc[2, 'filenames']
img_3 = val_dir + '/' + df_2.loc[3, 'filenames']

plt.figure(figsize=(10,10))


plt.subplot(2,2,1)
plt.imshow(plt.imread(img_0), cmap='gray')
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(plt.imread(img_1), cmap='gray')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(plt.imread(img_2), cmap='gray')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(plt.imread(img_3), cmap='gray')
plt.axis('off')
# End of Model Building
### ===================================================================================== ###
# Convert the Model from Keras to Tensorflow.js

!pip install tensorflowjs
# create a directory to store the model files
os.mkdir('tfjs_dir')

# convert to Tensorflow.js
import tensorflowjs as tfjs

tfjs.converters.save_keras_model(model, 'tfjs_dir')
# check the the directory containing the model is available
!ls
# view the files that make up the tensorflow.js model
os.listdir('tfjs_dir')
# Delete the image data directory we created to prevent a Kaggle error.
# Kaggle allows a max of 500 files to be saved.

shutil.rmtree('base_dir')
