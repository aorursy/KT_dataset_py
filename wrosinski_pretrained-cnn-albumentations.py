import gc

import glob

import os

import json

import matplotlib.pyplot as plt

import pprint



import cv2

import numpy as np

import pandas as pd



from joblib import Parallel, delayed

from tqdm import tqdm

from PIL import Image



%matplotlib inline



pd.options.display.max_rows = 128

pd.options.display.max_columns = 128
plt.rcParams['figure.figsize'] = (12, 9)
train = pd.read_csv('../input/train/train.csv')

test = pd.read_csv('../input/test/test.csv')

sample_submission = pd.read_csv('../input/test/sample_submission.csv')
labels_breed = pd.read_csv('../input/breed_labels.csv')

labels_state = pd.read_csv('../input/color_labels.csv')

labels_color = pd.read_csv('../input/state_labels.csv')
# Train files:

train_image_files = sorted(glob.glob('../input/train_images/*.jpg'))

train_metadata_files = sorted(glob.glob('../input/train_metadata/*.json'))

train_sentiment_files = sorted(glob.glob('../input/train_sentiment/*.json'))



print('num of train images files: {}'.format(len(train_image_files)))

print('num of train metadata files: {}'.format(len(train_metadata_files)))

print('num of train sentiment files: {}'.format(len(train_sentiment_files)))



# Test files:

test_image_files = sorted(glob.glob('../input/test_images/*.jpg'))

test_metadata_files = sorted(glob.glob('../input/test_metadata/*.json'))

test_sentiment_files = sorted(glob.glob('../input/test_sentiment/*.json'))



print('num of test images files: {}'.format(len(test_image_files)))

print('num of test metadata files: {}'.format(len(test_metadata_files)))

print('num of test sentiment files: {}'.format(len(test_sentiment_files)))
plt.rcParams['figure.figsize'] = (12, 9)

plt.style.use('ggplot')





print('train:')

# Images:

train_df_ids = train[['PetID']]

print(train_df_ids.shape)



train_df_imgs = pd.DataFrame(train_image_files)

train_df_imgs.columns = ['image_filename']

train_imgs_pets = train_df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])

train_df_imgs = train_df_imgs.assign(PetID=train_imgs_pets)

print(len(train_imgs_pets.unique()))



pets_with_images = len(np.intersect1d(train_imgs_pets.unique(), train_df_ids['PetID'].unique()))

print('fraction of pets with images: {:.3f}'.format(pets_with_images / train_df_ids.shape[0]))
print('test:')

# Images:

test_df_ids = test[['PetID']]

print(test_df_ids.shape)



test_df_imgs = pd.DataFrame(test_image_files)

test_df_imgs.columns = ['image_filename']

test_imgs_pets = test_df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])

test_df_imgs = test_df_imgs.assign(PetID=test_imgs_pets)

print(len(test_imgs_pets.unique()))



pets_with_images = len(np.intersect1d(test_imgs_pets.unique(), test_df_ids['PetID'].unique()))

print('fraction of pets with images: {:.3f}'.format(pets_with_images / test_df_ids.shape[0]))
import albumentations as A

import numpy as np

import keras



from keras import optimizers

from keras.applications.resnet50 import ResNet50, preprocess_input

from keras.applications import mobilenet

from keras.applications import mobilenet_v2



from keras.callbacks import *

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.layers import *

from keras.models import Model, load_model, save_model

from keras.preprocessing.image import array_to_img, img_to_array, load_img

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split







# Keras Data Generator for loading data from disk

# Based on: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class PetfinderDataGenerator(keras.utils.Sequence):



    def __init__(self, 

                 img_list, 

                 labels_list,

                 parser,

                 batch_size=16,

                 image_size=(224, 224),

                 n_channels=3,

                 n_classes=5,

                 shuffle=True):



        # List of images used for data loading

        self.img_list = img_list

        # List of corresponding labels

        self.labels_list = labels_list

        # Parser to use for image loading

        self.parser = parser

        # Batch size

        self.batch_size = batch_size

        # Size of image to be fed to the model

        self.image_size = image_size

        # Image channels

        self.n_channels = n_channels

        # Target number of classes

        self.n_classes = n_classes

        # Whether to shuffle image_list

        self.shuffle = shuffle



        self.on_epoch_end()



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.img_list) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[

            index * self.batch_size:(index + 1) * self.batch_size]



        # Find list of IDs

        img_list_temp = [self.img_list[k] for k in indexes]

        labels_list_temp = [self.labels_list[k] for k in indexes]



        # Generate data

        X, y = self.__data_generation(img_list_temp, labels_list_temp)



        return X, y



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.img_list))

        if self.shuffle is True:

            np.random.shuffle(self.indexes)



    def __data_generation(self, img_list_temp, label_list_temp):

        'Generates data containing batch_size samples'

        

        # Initialization

        X = np.empty((self.batch_size, *self.image_size, self.n_channels), dtype=np.float32)

        y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)



        # Generate data

        for i in range(len(img_list_temp)):



            img_temp = self.parser.load_image(img_list_temp[i])

            label_temp = label_list_temp[i]



            X[i], y[i] = img_temp, label_temp



        return X, y

    



# Helper class for image loading and augmentation

class PetfinderImageParser(object):

    

    def __init__(self,

                 preproc_func,

                 image_size=(224, 224),

                 transform=False,

                 resize=False, 

                 debug=False):

        

        self.image_size = image_size

        self.preproc_func = preproc_func

        self.transform = transform

        self.resize = resize

        self.debug = debug

    

        

    def load_image(self, img_filename, preprocess=True):

        

        image = cv2.imread(img_filename)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        

        if self.transform:

            image = self.transform(image=image)['image']

        

        if image.shape[:2] != self.image_size and self.resize:

            image = cv2.resize(image, self.image_size)

            

        if preprocess:

            image = self.preproc_func(image)

        

        return image

    



def PetfinderResNet(input_size, num_classes=5):

    

    base_model = ResNet50(

        input_shape=input_size, 

        include_top=False,

        weights='imagenet')

    

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(256)(x)

    x = PReLU()(x)

    x = Dense(256)(x)

    x = PReLU()(x)

    x = Dense(num_classes, activation='softmax')(x)

    

    model = Model(inputs=base_model.input, outputs=x)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    

    return model





def PetfinderMobileNet(input_size, num_classes=5):

    

    base_model = mobilenet.MobileNet(

        input_shape=input_size, 

        include_top=False,

        weights='imagenet')

    

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(256)(x)

    x = PReLU()(x)

    x = Dense(256)(x)

    x = PReLU()(x)

    x = Dense(num_classes, activation='softmax')(x)

    

    model = Model(inputs=base_model.input, outputs=x)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    

    return model





def PetfinderMobileNetV2(input_size, num_classes=5):

    

    base_model = mobilenet_v2MobileNetV2(

        input_shape=input_size, 

        include_top=False,

        weights='imagenet')

    

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(256)(x)

    x = PReLU()(x)

    x = Dense(256)(x)

    x = PReLU()(x)

    x = Dense(num_classes, activation='softmax')(x)

    

    model = Model(inputs=base_model.input, outputs=x)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    

    return model
# Debug parameter for training only on subset of data 

# for quick experiments

DEBUG = True





# Unique IDs from train and test:

train_pet_ids = train.PetID.unique()

test_pet_ids = test.PetID.unique()



if DEBUG:

    train_pet_ids = train_pet_ids[:32]

    test_pet_ids = test_pet_ids[:16]



    

print(len(train_pet_ids), len(test_pet_ids))



# Merge AdoptionSpeed (target columns) onto images DF:

train_df_imgs = train_df_imgs.merge(train[['PetID', 'AdoptionSpeed']], how='left', on='PetID')
# Train/valid split based on PetID

tr_ids, valid_ids = train_test_split(train_pet_ids, test_size=0.2, random_state=1337)



tr_df = train_df_imgs.loc[train_df_imgs.PetID.isin(tr_ids)].reset_index(drop=True)

valid_df = train_df_imgs.loc[train_df_imgs.PetID.isin(valid_ids)].reset_index(drop=True)



print(tr_df.shape, valid_df.shape)



assert len(set(tr_df.PetID.unique()).intersection(valid_df.PetID.unique())) == 0
# Cropping borders:

image_crop_pad = 64

# Batch size for model training:

batch_size = 64



# Image size for training:

image_size = (128, 128)

input_size = image_size + (3,)

print(image_size, input_size)



# Choose which model to use

model_touse = 'MobileNet'

assert model_touse in ['MobileNet', 'MobileNetV2', 'ResNet50']



# Choose proper preprocessing function for model

if model_touse == 'MobileNet':

    preproc_func = mobilenet.preprocess_input

elif model_touse == 'MobileNetV2':

    preproc_func = mobilenet_v2.preprocess_input

elif model_touse == 'ResNet50':

    preproc_func = preprocess_input

    



train_steps_per_epoch = tr_df.shape[0] // batch_size





# Define augmentations for training & validation:

train_aug = A.Compose([

    A.HorizontalFlip(),

    A.RandomRotate90(),

    # First resize to retain some variability in random crops

    # but avoid situations where pet is completely removed from the crop

    A.Resize(image_size[0] + image_crop_pad,

             image_size[1] + image_crop_pad),

    A.RandomScale(0.25),

    A.RandomCrop(image_size[0], image_size[1])

])



valid_aug = A.Compose([

    # Do the same for valid

    A.Resize(image_size[0] + image_crop_pad,

             image_size[1] + image_crop_pad),

    A.CenterCrop(image_size[0], image_size[1])

])





# Initialize training data generator:

filenames_tr = tr_df.image_filename.values

y_tr = tr_df.AdoptionSpeed.values

y_tr = to_categorical(y_tr)

print('train set shapes:')

print(filenames_tr.shape, y_tr.shape)



tr_parser = PetfinderImageParser(preproc_func, image_size, train_aug)

tr_datagen = PetfinderDataGenerator(

    filenames_tr, y_tr, tr_parser, 

    image_size=image_size,

    batch_size=batch_size)





# Initialize validation data generator:

filenames_valid = valid_df.image_filename.values

y_valid = valid_df.AdoptionSpeed.values

y_valid = to_categorical(y_valid)

print('valid set shapes:')

print(filenames_valid.shape, y_valid.shape)



valid_parser = PetfinderImageParser(preproc_func, image_size, valid_aug)

valid_datagen = PetfinderDataGenerator(

    filenames_valid, y_valid, valid_parser, 

    image_size=image_size, 

    batch_size=batch_size)
plt.style.use('default')

plt.rcParams['figure.figsize'] = (16, 12)





N_COLS = 3

N_ROWS = 3



fig, ax = plt.subplots(N_COLS, N_ROWS)



for c in range(N_COLS):

    for r in range(N_ROWS):

        ridx = np.random.randint(0, len(tr_df))

        img_row = tr_df.iloc[ridx, :]

        img_filename = img_row['image_filename']

        pet_id = img_row['PetID']

        pet_label = img_row['AdoptionSpeed']

        

        # Explicitly set preprocess to False for image inspection

        # Should be set to True (it is by default) for model training

        image = tr_parser.load_image(img_filename, preprocess=False)

        ax[c, r].imshow(image)

        ax[c, r].set_title('ID: {}, label: {}'.format(pet_id, pet_label), size=12)
plt.style.use('default')

plt.rcParams['figure.figsize'] = (16, 12)





N_COLS = 3

N_ROWS = 3



fig, ax = plt.subplots(N_COLS, N_ROWS)



for c in range(N_COLS):

    for r in range(N_ROWS):

        ridx = np.random.randint(0, len(valid_df))

        img_row = valid_df.iloc[ridx, :]

        img_filename = img_row['image_filename']

        pet_id = img_row['PetID']

        pet_label = img_row['AdoptionSpeed']

        

        image = valid_parser.load_image(img_filename, preprocess=False)

        ax[c, r].imshow(image)

        ax[c, r].set_title('ID: {}, label: {}'.format(pet_id, pet_label), size=12)
plt.rcParams['figure.figsize'] = (10, 10)



# train datagen:

X_tr_temp, y_tr_temp = tr_datagen.__getitem__(0)



# valid datagen:

X_valid_temp, y_valid_temp = valid_datagen.__getitem__(0)





fig, ax = plt.subplots(1, 2)



ax[0].imshow(X_tr_temp[0])

ax[0].set_title('train:')



ax[1].imshow(X_valid_temp[0])

ax[1].set_title('valid:')



# Here we can see that when loaded for training by default images are preprocessed with ResNet function.
if model_touse == 'MobileNet':

    model = PetfinderMobileNet(input_size)

elif model_touse == 'MobileNetV2':

    model = PetfinderMobileNetV2(input_size)

elif model_touse == 'ResNet50':

    model = PetfinderResNet(input_size)

    

print(model.summary())
model_checkpoint = ModelCheckpoint(

    'petfinder_cnn.h5' ,monitor='val_loss', mode='min',

    save_best_only=True, save_weights_only=True, verbose=1)

reduce_lr = ReduceLROnPlateau(

    monitor='val_loss',

    mode='min',

    factor=0.5, 

    patience=5, 

    min_lr=0.0001, 

    verbose=1)





model.fit_generator(

    generator=tr_datagen,

    steps_per_epoch=train_steps_per_epoch,

    validation_data=valid_datagen,

    epochs=1,

    use_multiprocessing=True,

    workers=2,

    callbacks=[model_checkpoint, reduce_lr])