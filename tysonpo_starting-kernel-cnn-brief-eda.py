import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import json

import os

import re

import seaborn as sns

from collections import Counter

import random



# -- Modeling --

import tensorflow as tf

import keras

from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator



# install Keras-Applications

!pip install Keras-Applications

from keras_applications.imagenet_utils import _obtain_input_shape

from keras import backend as K

from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, Dense

from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D

from keras.models import Model, Sequential

from keras.engine.topology import get_source_inputs

from keras.utils import get_file

from keras.utils import layer_utils



import warnings  

warnings.filterwarnings('ignore')



def seed_everything(seed = 42):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.random.set_seed(seed)

    

seed_everything()
data = pd.read_csv("../input/bike-ads-images-prices-specifications/combined_price-only.csv")

data.shape
data.head()
data.info()
data.describe()
# choose indices for train-test split

n = len(data)

train_idx = np.random.choice(range(n), size=int(0.8*n), replace=False)

test_idx = [i for i in range(n) if i not in train_idx]





print("Number of training examples:", len(train_idx)) 

print("Number of testing examples:", len(test_idx))

print("Number of examples that appear in both (should be zero):", len(set(train_idx).intersection(set(test_idx))))
# make train-test split

train_label_df = data.loc[train_idx,:]

test_label_df = data.loc[test_idx,:]



# save as csv (optional)

train_label_df.to_csv("./df_train.csv", index=False)

test_label_df.to_csv("./df_test.csv", index=False)
sq1x1 = "squeeze1x1"

exp1x1 = "expand1x1"

exp3x3 = "expand3x3"

relu = "relu_"



WEIGHTS_PATH = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels.h5"

WEIGHTS_PATH_NO_TOP = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5"



# Modular function for Fire Node



def fire_module(x, fire_id, squeeze=16, expand=64):

    s_id = 'fire' + str(fire_id) + '/'



    if K.image_data_format() == 'channels_first':

        channel_axis = 1

    else:

        channel_axis = 3

    

    x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)

    x = Activation('relu', name=s_id + relu + sq1x1)(x)



    left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)

    left = Activation('relu', name=s_id + relu + exp1x1)(left)



    right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)

    right = Activation('relu', name=s_id + relu + exp3x3)(right)



    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')

    return x





# Original SqueezeNet from paper.



def SqueezeNet(include_top=True, weights='imagenet',

               input_tensor=None, input_shape=None,

               pooling=None,

               classes=1000):

    """Instantiates the SqueezeNet architecture.

    """

        

    if weights not in {'imagenet', None}:

        raise ValueError('The `weights` argument should be either '

                         '`None` (random initialization) or `imagenet` '

                         '(pre-training on ImageNet).')



    if weights == 'imagenet' and classes != 1000:

        raise ValueError('If using `weights` as imagenet with `include_top`'

                         ' as true, `classes` should be 1000')





    input_shape = _obtain_input_shape(input_shape,

                                      default_size=227,

                                      min_size=48,

                                      data_format=K.image_data_format(),

                                      require_flatten=include_top)



    if input_tensor is None:

        img_input = Input(shape=input_shape)

    else:

        if not K.is_keras_tensor(input_tensor):

            img_input = Input(tensor=input_tensor, shape=input_shape)

        else:

            img_input = input_tensor





    x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(img_input)

    x = Activation('relu', name='relu_conv1')(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)



    x = fire_module(x, fire_id=2, squeeze=16, expand=64)

    x = fire_module(x, fire_id=3, squeeze=16, expand=64)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)



    x = fire_module(x, fire_id=4, squeeze=32, expand=128)

    x = fire_module(x, fire_id=5, squeeze=32, expand=128)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)



    x = fire_module(x, fire_id=6, squeeze=48, expand=192)

    x = fire_module(x, fire_id=7, squeeze=48, expand=192)

    x = fire_module(x, fire_id=8, squeeze=64, expand=256)

    x = fire_module(x, fire_id=9, squeeze=64, expand=256)

    

    if include_top:

        # It's not obvious where to cut the network... 

        # Could do the 8th or 9th layer... some work recommends cutting earlier layers.

    

        x = Dropout(0.5, name='drop9')(x)



        x = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)

        x = Activation('relu', name='relu_conv10')(x)

        x = GlobalAveragePooling2D()(x)

        x = Activation('softmax', name='loss')(x)

    else:

        if pooling == 'avg':

            x = GlobalAveragePooling2D()(x)

        elif pooling=='max':

            x = GlobalMaxPooling2D()(x)

        elif pooling==None:

            pass

        else:

            raise ValueError("Unknown argument for 'pooling'=" + pooling)



    # Ensure that the model takes into account

    # any potential predecessors of `input_tensor`.

    if input_tensor is not None:

        inputs = get_source_inputs(input_tensor)

    else:

        inputs = img_input



    model = Model(inputs, x, name='squeezenet')



    # load weights

    if weights == 'imagenet':

        if include_top:

            weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels.h5',

                                    WEIGHTS_PATH,

                                    cache_subdir='models')

        else:

            weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5',

                                    WEIGHTS_PATH_NO_TOP,

                                    cache_subdir='models')

            

        model.load_weights(weights_path)

        if K.backend() == 'theano':

            layer_utils.convert_all_kernels_in_model(model)



    return model



img_width = 227

img_height = 227



model = Sequential()

squeezenet = SqueezeNet(weights='imagenet', include_top=False, input_shape = (img_width, img_height, 3), pooling=None)

model.add(squeezenet)



# "Top layer" adapted for regression

model.add(Dropout(0.5))

model.add(Convolution2D(1000, (1, 1), padding='valid'))

model.add(Activation('relu'))

model.add(GlobalAveragePooling2D())

model.add(Dense(1, activation='linear'))



model.summary()
# path to images

img_dir = "../input/bike-ads-images-prices-specifications/images"



# append .jpg to id

def append_ext(fn):

    return str(fn) + ".jpg"

train_label_df["ID"] = train_label_df["ID"].apply(append_ext)

test_label_df["ID"] = test_label_df["ID"].apply(append_ext)



# parameters -- increase these to improve performance

bs = 32 # batch size

epochs = 8 

steps_per_epoch = 50 

validation_steps = 50



# ------------ DATA GENERATORS ------------

# All images will be rescaled by 1./255



# Training set: no data augmentation

# train_datagen = ImageDataGenerator(rescale=1./255) 



# Training set: with data augmentation

train_datagen = ImageDataGenerator(rescale = 1./255, 

                                   rotation_range=30, 

                                   width_shift_range = 0.2, 

                                   height_shift_range=0.2,

                                   shear_range=0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



train_generator = train_datagen.flow_from_dataframe(dataframe=train_label_df, 

                                                    directory=img_dir, 

                                                    x_col="ID", # id

                                                    y_col="Price", # score

                                                    class_mode="raw", 

                                                    target_size=(img_width, img_height), 

                                                    batch_size=bs)



# Testing set

test_datagen = ImageDataGenerator(rescale=1./255)



validation_generator = test_datagen.flow_from_dataframe(dataframe=test_label_df, 

                                                    directory=img_dir, 

                                                    x_col="ID", # id

                                                    y_col="Price", # score

                                                    class_mode="raw", 

                                                    target_size=(img_width, img_height), 

                                                    batch_size=bs)



# Amount+dimensions of generated data

for data_batch, labels_batch in train_generator:

    print('data batch shape:', data_batch.shape)

    print('labels batch shape:', labels_batch.shape)

    break

# ----------------------------------------



# Train the model

model.compile(loss='mse', optimizer=optimizers.RMSprop(lr=0.0001), metrics=['mae', 'mse']) # lr=0.001 to 0.01 seem to work well 

history = model.fit_generator(train_generator,

                              steps_per_epoch=steps_per_epoch,

                              epochs=epochs,

                              validation_data=validation_generator,

                              validation_steps=validation_steps)
hist_dict = model.history.history



fig, ax = plt.subplots(1,2,figsize=(10,4))



# MAE

plt.sca(ax[0])

plt.plot(hist_dict["mae"], label="training")

plt.plot(hist_dict["val_mae"], label="validation")

plt.xlabel("Epochs")

plt.ylabel("Mean Absolute Error")

plt.legend()



# MSE

plt.sca(ax[1])

plt.plot(hist_dict["mse"], label="training")

plt.plot(hist_dict["val_mse"], label="validation")

plt.xlabel("Epochs")

plt.ylabel("Mean Squared Error")

plt.legend()



plt.tight_layout()

plt.show()
(test_label_df["Price"]-train_label_df["Price"].median()).abs().mean()
site = "ebay"



ignore = ["Price now", "Price was", "Price", "ID", "Condition", "Seller notes", "Title", "Product URL", "Photo URL", "UPC", "MPN"]

file = "../input/bike-ads-images-prices-specifications/data_%s.json" % site

attributes = []

with open(file,"r") as f:

    for line in f:

        x = json.loads(line)

        attributes.extend([y for y in x.keys() if y not in ignore])



len(set(attributes))
common_attributes = Counter(attributes).most_common(25) # top 25 most common 

xy = list(zip(*common_attributes)) # getting ready to plot

fig = plt.figure(figsize=(16,4))

chart = sns.barplot(x=list(xy[0]), y=list(xy[1]))

chart.set_xticklabels(chart.get_xticklabels(), rotation=60, horizontalalignment='right', fontsize=14)

plt.show()
# sites = ["ebay","bike_exchange"] # uncomment this to loop over both sites

sites = ["ebay"]



POUND_TO_DOLLAR_RATIO = 1.25 # as of June 2020



IDs = []

prices = []

for site in sites:

    file = "../input/bike-ads-images-prices-specifications/data_%s.json" % site

    

    with open(file,"r") as f:

        for line in f:

            x = json.loads(line)

            

            if site == "bike_exchange":

                price_key = "Price now" 

            else:

                price_key = "Price"

            

            price_US = re.findall("[\$]{1}[\d,]+\.?\d{0,2}",x[price_key])

            price_UK = re.findall("[\u00a3]{1}[\d,]+\.?\d{0,2}",x[price_key])



            if price_US:

                float_price = float(price_US[0].replace("$","").replace(",",""))

            elif price_UK:

                float_price = POUND_TO_DOLLAR_RATIO * float(price_UK[0].replace("\u00a3","").replace(",",""))



            IDs.append(int(x["ID"]))

            prices.append(round(float_price, 2))



# plotting

plt.hist(prices, bins="auto")

plt.show()