# The following was run on colab.  It would need minor modifications to run it here on kaggle:

def prep_data():

    df=pd.read_csv('bee_data.csv', 

                    index_col=False,  

                    parse_dates={'datetime':[1,2]},

                    dtype={'subspecies':'category', 'health':'category','caste':'category'})



    out_dir = '/content/bees'

    shuffle = np.random.randint(df.shape[0], size=(df.shape[0]))

    test_size = 500

    test_data = shuffle[:test_size]

    validate_data = shuffle[test_size:(test_size*2)]

    train_data = shuffle[(test_size*2):]



    logprint("input sizes:")

    logprint("  test = "+str(test_data.shape[0]))

    logprint("  valid = "+str(validate_data.shape[0]))

    logprint("  train = "+str(train_data.shape[0]))

    subfolders = ['test','validate','train']

    labels = []

    categories = {}



    #convert Varroa to few varrao:

    def convert(rec):

      if rec['health'] == 'Varroa, Small Hive Beetles':

        return 'few varrao, hive beetles' 

      else:

        return rec['health']

    df['health'] = df.apply(lambda x: convert(x), axis=1)



    #delete the missing queens:

    missing_queens = df[df.health == 'missing queen'].index

    logprint("num queens: " + str(missing_queens.shape[0]))

    df = df.drop(missing_queens)



    #create the subfolders and build the categories dictionary:

    count = 0

    for health in df.health.unique():

      category = "category"+str(count)

      categories[health] = category

      labels.append(health)

      for subfolder in subfolders:

        dirname = os.path.join(os.path.join(out_dir, subfolder), category)

        if not os.path.exists(dirname):

          os.makedirs(dirname)

      count += 1



    for key, value in categories.items():

      logprint("categories["+key+"] = "+value)



    #initialize a dictionary to store the counts of each sub-subfolder:

    counts = {}

    for s in range(len(subfolders)):

      counts[subfolders[s]] = {}

      for health, category in categories.items():

        counts[subfolders[s]][category] = 0



    count = 0

    for filename in os.listdir(base_dir):

      try:

        #for the missing queens the dataframe lookup will throw IndexError:

        health = df[df.file == filename].health.iloc[0]

        category = categories[health]



        fromfile = os.path.join(base_dir, filename)

        if count < test_size:

          subfolder = subfolders[0]

        elif count < 2*test_size:

          subfolder = subfolders[1]

        else:

          subfolder = subfolders[2]

        counts[subfolder][category] += 1

        todir = os.path.join(out_dir, os.path.join(subfolder, category))

        tofile = os.path.join(todir, filename)

        shutil.copyfile(fromfile, tofile)

        count += 1

      except IndexError:

        None



    totals = {}

    for subfolder, cats in counts.items():

      if subfolder not in totals.keys():

        totals[subfolder] = 0

      for category, num in cats.items():

        logprint("counts["+subfolder+"]["+category+"]: "+str(num))

        totals[subfolder] += counts[subfolder][category]

    logprint("totals:")

    logprint(totals)
import numpy as np

import pandas as pd

import random, datetime, os, shutil, math

import matplotlib.pyplot as plt

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from keras import layers

from keras import models

from keras import optimizers

import os



image_size = (150, 150)



base_filename = '../input/modelcache/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

val_acc_filename = 'hbhc-simple-val_acc.h5'

val_loss_filename = 'hbhc-simple-val_loss.h5'

save_filename = 'hbhc-simple-model.h5'

hist_filename = 'hbhc-simple-hist.csv'



#images in:

input_dir = '../input/honeybees-simplified/bees-simple/bees/'

train_dir = input_dir + "train"

test_dir = input_dir + "test"

validate_dir = input_dir + "validate"



log_filename = "hbhc_log.txt"

log_file = open(log_filename, "a")

  

# timestamp and then write the msg to both the console and the log file:

def logprint(msg):

  msg_str = "["+str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+"] "+str(msg)

  print(msg_str)

  log_file = open(log_filename, "a")

  log_file.write(msg_str+"\n")

  log_file.close()



logprint("Reopened log file "+log_filename)



#display a sample of bee photos in an auto-sized grid:

def show_bees(bzz):

  numbees = len(bzz)

  if numbees == 0:

    return None

  rows = int(math.sqrt(numbees))

  cols = (numbees+1)//rows

  f, axs = plt.subplots(rows, cols)

  fig = 0

  for b in bzz:

    img = image.load_img(b)

    row = fig // cols

    col = fig % cols

    axs[row, col].imshow(img)

    fig += 1

  plt.show()

  
#show some sample images:

dir_name = os.path.join(test_dir,"category0")

all_images = [os.path.join(dir_name, fname) for fname in os.listdir(dir_name)]

show_bees(all_images[:6])

# This is F. Chollet's VGG16 pretrained convnet, slightly modified by yours truly to adapt it to kaggle.



# -*- coding: utf-8 -*-

'''VGG16 model for Keras.

# Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

'''

from __future__ import print_function



import numpy as np

import warnings



from keras.models import Model

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Input

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import GlobalMaxPooling2D

from keras.layers import GlobalAveragePooling2D

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras import backend as K

from keras.applications.imagenet_utils import decode_predictions

from keras.applications.imagenet_utils import preprocess_input

#from keras.applications.imagenet_utils import _obtain_input_shape

from keras_applications.imagenet_utils import _obtain_input_shape

from keras.engine.topology import get_source_inputs





#WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

#WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'



WEIGHTS_PATH_NO_TOP = base_filename



def VGG16(weights='imagenet',

          input_tensor=None, input_shape=None,

          pooling=None,

          classes=1000):

    """Instantiates the VGG16 architecture.

    Optionally loads weights pre-trained

    on ImageNet. Note that when using TensorFlow,

    for best performance you should set

    `image_data_format="channels_last"` in your Keras config

    at ~/.keras/keras.json.

    The model and the weights are compatible with both

    TensorFlow and Theano. The data format

    convention used by the model is the one

    specified in your Keras config file.

    # Arguments



        weights: one of `None` (random initialization)

            or "imagenet" (pre-training on ImageNet).

        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)

            to use as image input for the model.

        input_shape: optional shape tuple, only to be specified

            if `include_top` is False (otherwise the input shape

            has to be `(224, 224, 3)` (with `channels_last` data format)

            or `(3, 224, 244)` (with `channels_first` data format).

            It should have exactly 3 inputs channels,

            and width and height should be no smaller than 48.

            E.g. `(200, 200, 3)` would be one valid value.

        pooling: Optional pooling mode for feature extraction

            when `include_top` is `False`.

            - `None` means that the output of the model will be

                the 4D tensor output of the

                last convolutional layer.

            - `avg` means that global average pooling

                will be applied to the output of the

                last convolutional layer, and thus

                the output of the model will be a 2D tensor.

            - `max` means that global max pooling will

                be applied.

        classes: optional number of classes to classify images

            into, only to be specified if `include_top` is True, and

            if no `weights` argument is specified.

    # Returns

        A Keras model instance.

    # Raises

        ValueError: in case of invalid argument for `weights`,

            or invalid input shape.

    """

    if weights not in {'imagenet', None}:

        raise ValueError('The `weights` argument should be either '

                         '`None` (random initialization) or `imagenet` '

                         '(pre-training on ImageNet).')





    # Determine proper input shape

    input_shape = _obtain_input_shape(input_shape,

                                      default_size=224,

                                      min_size=48,

                                      require_flatten=True,

                                      data_format=K.image_data_format())



    if input_tensor is None:

        img_input = Input(shape=input_shape)

    else:

        if not K.is_keras_tensor(input_tensor):

            img_input = Input(tensor=input_tensor, shape=input_shape)

        else:

            img_input = input_tensor

    # Block 1

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)



    # Block 2

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)



    # Block 3

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)



    # Block 4

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)



    # Block 5

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)



    if pooling == 'avg':

        x = GlobalAveragePooling2D()(x)

    elif pooling == 'max':

        x = GlobalMaxPooling2D()(x)



    # Ensure that the model takes into account

    # any potential predecessors of `input_tensor`.

    if input_tensor is not None:

        inputs = get_source_inputs(input_tensor)

    else:

        inputs = img_input

    # Create model.

    model = Model(inputs, x, name='vgg16')



    # load weights



    if weights == 'imagenet':

        model.load_weights(WEIGHTS_PATH_NO_TOP)

            

        if K.backend() == 'theano':

            layer_utils.convert_all_kernels_in_model(model)



        if K.image_data_format() == 'channels_first':

            if K.backend() == 'tensorflow':

                warnings.warn('You are using the TensorFlow backend, yet you '

                              'are using the Theano '

                              'image data format convention '

                              '(`image_data_format="channels_first"`). '

                              'For best performance, set '

                              '`image_data_format="channels_last"` in '

                              'your Keras config '

                              'at ~/.keras/keras.json.')

    return model



def build_model():

  conv_base = VGG16(weights='imagenet', \

                    input_shape=(image_size[0], image_size[1], 3))

  model = models.Sequential()

  model.add(conv_base)

  model.add(layers.Flatten())

  model.add(layers.Dense(256, activation='relu'))

  model.add(layers.Dropout(rate=0.5))

  model.add(layers.Dense(4, activation='softmax'))



  conv_base.trainable = True

  set_trainable = False

  for layer in conv_base.layers:

      if layer.name == 'block4_conv1':

          set_trainable = True

      if set_trainable:

          layer.trainable = True

      else:

          layer.trainable = False



  model.compile(loss='categorical_crossentropy',

                optimizer=optimizers.RMSprop(lr=1e-5),

                metrics=['acc'])

  conv_base.summary()

  return model

  

# return a list of callbacks for training:

def callbacks():

  ckpt_val_acc = ModelCheckpoint(val_acc_filename, monitor='val_acc', save_best_only=True)

  ckpt_val_loss = ModelCheckpoint(val_loss_filename, monitor='val_loss', save_best_only=True)

  ckpt_model = ModelCheckpoint(save_filename, save_best_only=False)

  save_hist = CSVLogger(hist_filename, separator=',', append=True)

  return [ckpt_val_acc,ckpt_val_loss,ckpt_model,save_hist]



train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest')

train_flow = train_datagen.flow_from_directory(

        train_dir,

        target_size=image_size,

        batch_size=20,

        class_mode='categorical')

validation_datagen = ImageDataGenerator(rescale=1./255)

validate_flow = validation_datagen.flow_from_directory(

        validate_dir,

        target_size=image_size,

        batch_size=20,

        class_mode='categorical')



if os.path.isfile(save_filename):

    # restore saved model from prior run:

    model = models.load_model(save_filename)

    logprint("loaded model from cache: "+save_filename)

else:

    model = build_model()

    logprint("no cache found: "+save_filename)

    

model.summary()

logprint("training start (epochs limited to 1 for demo only)")

history = model.fit_generator(

      train_flow,

      steps_per_epoch=100,

      epochs=1,

      validation_data=validate_flow,

      validation_steps=50,

      callbacks=callbacks())

logprint("training complete")
