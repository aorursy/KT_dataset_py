# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os)

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import keras

import numpy as np

import keras

from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.preprocessing.image import load_img

from keras import regularizers

from keras.models import Model

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

from keras.applications import Xception 

from keras.applications import imagenet_utils

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.callbacks.callbacks import EarlyStopping

from keras.callbacks.callbacks import ReduceLROnPlateau

from keras.callbacks.callbacks import ModelCheckpoint

import pydot

from IPython.display import SVG

from keras.utils import plot_model

from keras.utils import to_categorical

from random import randint

from PIL import Image

from sklearn.utils import class_weight

from sklearn.model_selection import train_test_split



import keras.backend as K

K.set_image_data_format('channels_last')

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow



%matplotlib inline
train_data = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')

test_data = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')

print(train_data.to_numpy().shape)

train_data.describe()



train_data.head()

train_sum = train_data[['healthy', 'multiple_diseases', 'rust', 'scab']].sum(axis=0)



print('Healthy: ' + str(train_sum[0]))

print('Multiple Diseases: ' + str(train_sum[1]))

print('Rust: ' + str(train_sum[2]))

print('Scab: ' + str(train_sum[3]))

print(test_data.shape)

train_sum.head()

test_data.head()

HEALTHY = 0

MULTIPLE_DISEASES = 1

RUST = 2

SCAB = 3



def transform(row):

    if row.healthy == 1:

        return HEALTHY

    elif row.multiple_diseases == 1:

        return MULTIPLE_DISEASES

    elif row.rust == 1:

        return RUST

    else:

        return SCAB

    

train_data['class'] = train_data.apply(lambda row: transform(row), axis=1)

train_classed = train_data[['image_id', 'class']]

class_weights = class_weight.compute_class_weight(

    'balanced',

    train_classed['class'].unique(),

    train_classed[['class']].to_numpy().reshape(-1)

)

print(train_classed['class'].unique())

print(train_classed[['class']].to_numpy().reshape(-1))

class_weights = {

    HEALTHY: class_weights[HEALTHY],

    MULTIPLE_DISEASES: class_weights[MULTIPLE_DISEASES],

    RUST: class_weights[RUST],

    SCAB: class_weights[SCAB],

}

print('Weights')

print('Healthy: ' + str(class_weights[HEALTHY]))

print('Multiple Diseases: ' + str(class_weights[MULTIPLE_DISEASES]))

print('Rust: ' + str(class_weights[RUST]))

print('Scab: ' + str(class_weights[SCAB]))

print(class_weights)

train_classed.head()
ROOT = '/kaggle/input/plant-pathology-2020-fgvc7/images/'

IMAGE_DIM_X = 299

IMAGE_DIM_Y = 299

inception_preprocess = preprocess_input



def get_image_path(image_id):

    return ROOT + image_id + '.jpg'



def generate_labels(train_data, index):

    row = train_data.iloc[index]

    ret = ""

    if row.healthy == 1:

        ret = "Healthy"

    elif row.multiple_diseases == 1:

        ret = "Mutliple Diseases"

    elif row.rust == 1:

        ret = "Rust"

    elif row.scab == 1:

        ret = "Scab"

    else:

        raise ValueError("Incorrect leaf category")

    return str(index) + ":" + ret



def show_sample_images(start_index=0, count=10):

    """

    Shows next 10 sample images with labels

    """

    for i in range(count):

        index = i + start_index

        img_path = get_image_path(train_data.iloc[index].image_id)

        img = load_img(img_path)

        np_img = img_to_array(img).astype(int)

        fig = plt.figure()

        plt.imshow(np_img)

        plt.title(generate_labels(train_data, index), fontsize=10)

        plt.plot()

    plt.plot()
# show_sample_images(140, 40)



img_path = get_image_path(train_data.iloc[0].image_id)

img = load_img(img_path)

np_img = img_to_array(img).astype(int)

np_img.shape
def image_walker(input_data, image_name_col, label_cols, is_test=False):

    """

    Creates an ImageDataGenerator which serves two purposes:

    1. Loads images from a predefined directory and prepares batches for Gradient Descent.

    2. Performs data augmentation on those images.

    

    Parameters

    ----------

    input_paths : Dataframe containing a columns "paths" representing different paths from which an image

    is expected to be loaded.

    It is assumed that all the images are in the input/plant-pathalogy-2020-fgvc7/images directory

    

    Returns

    -------

    iterator to the created ImageDataGenerator

    """

    # TODO: Add Data Agumentation

    def append_extension(extension):

        return extension +".jpg"

    input_data_with_ext = input_data.copy()

    input_data_with_ext[image_name_col] = input_data_with_ext[image_name_col].apply(append_extension)

    

    data_generator = ImageDataGenerator(

        preprocessing_function=preprocess_input,

        rotation_range=20,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        fill_mode='nearest'

    )

    

    iterator = data_generator.flow_from_dataframe(

        input_data_with_ext, 

        directory=ROOT, 

        x_col=image_name_col, 

        y_col=label_cols,                                

        target_size=(IMAGE_DIM_X, IMAGE_DIM_Y),

        seed=42, 

        class_mode='raw',

        batch_size=16,

        shuffle=False,

        subset='training'

    )

    

    if is_test:

        data_generator = ImageDataGenerator(

            preprocessing_function=preprocess_input)

        def full_path(image_id):

            return ROOT + image_id 

        input_data_with_ext[image_name_col] = input_data_with_ext[image_name_col].apply(full_path)

        iterator = data_generator.flow_from_dataframe(

            input_data_with_ext, 

            directory=None, 

            x_col=image_name_col,  

            target_size=(IMAGE_DIM_X, IMAGE_DIM_Y),

            seed=42, 

            class_mode=None,

            batch_size=1,

            subset='validation'

        )

    return iterator

    
print("[INFO] loading {}...".format("inception"))

base_model = InceptionResNetV2(

    weights='imagenet', 

    include_top=False, 

    input_shape=(IMAGE_DIM_X, IMAGE_DIM_Y, 3))
CLASSES = 4

X = base_model.output

X = GlobalAveragePooling2D(name='avg_pool')(X)

predictions = Dense(CLASSES, activation='softmax')(X)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy']

             )
# Train Data generation

# X_train, X_test = train_test_split(train_classed, random_state=42, test_size=0.01)



# print('X_train shape: ' + str(X_train.shape))

# print('X_test shape: ' + str(X_test.shape))



X_train = train_classed



X_train.head()
early = EarlyStopping(monitor='accuracy', min_delta=0.1, patience=5, verbose=0, mode='auto', restore_best_weights=True)



reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.0001, mode='auto')



model_checkpoint = ModelCheckpoint('incepresnetv2_ol.model', monitor='loss', save_best_only=True)
EPOCHS = 50

BATCH_SIZE = 16

STEPS_PER_EPOCH = 114

MODEL_FILE = 'filename_ol_dense.model'

history = model.fit_generator(

    image_walker(X_train, 'image_id', 'class'),

    shuffle=True,

    epochs=EPOCHS,

    steps_per_epoch=STEPS_PER_EPOCH,

    callbacks=[model_checkpoint, reduce_lr],

#     validation_data=image_walker(X_test, 'image_id', 'class'),

#     validation_steps=18,

    class_weight=class_weights

)

  

model.save(MODEL_FILE)
# eval_data_generator = image_walker(X_test, 'image_id', 'class')

# preds = model.evaluate_generator(eval_data_generator)

# print ("Loss val = " + str(preds[0]))

# print ("Test Accuracy val = " + str(preds[1]))
image_ids = test_data.to_numpy().flatten()

print(image_ids)

train_images = np.empty([image_ids.shape[0], IMAGE_DIM_X, IMAGE_DIM_Y, 3])



for i in range(image_ids.shape[0]):

    train_images[i] = np.uint8(Image.open(ROOT + image_ids[i] + '.jpg').resize((IMAGE_DIM_X, IMAGE_DIM_Y)))



print(train_images[0])
test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

test_predictions = model.predict_generator(

    test_generator.flow(train_images, batch_size=1, shuffle=False), 

    steps=1821, # Count of test samples,

    verbose=1

    )
result = pd.DataFrame()

result['image_id'] = pd.Series(image_ids)

result['healthy'] = pd.Series(test_predictions[:, 0])

result['multiple_diseases'] = pd.Series(test_predictions[:, 1])

result['rust'] = pd.Series(test_predictions[:, 2])

result['scab'] = pd.Series(test_predictions[:, 3])

result.head()
result.to_csv('submission.csv', index=False)