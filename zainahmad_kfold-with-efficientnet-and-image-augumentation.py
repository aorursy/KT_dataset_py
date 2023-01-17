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
# installing efficientnet

!pip install efficientnet
from kaggle_datasets import KaggleDatasets

import tensorflow as tf

import tensorflow_addons as tfa

from tensorflow.keras.layers import Dense , Activation , GlobalAveragePooling2D , Dropout ,Input

from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing import image

import efficientnet.tfkeras as efn

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold ,KFold

from PIL import Image

import io

import cv2

import pandas as pd

import numpy as np

import re
IMG_DIMS = 64

CHANNELS = 3

BATCH_SIZE = 32

SEED = 42

SPLITS = 4
def parse_TFR_data_labelled(sample):

    features = {

      'image': tf.io.FixedLenFeature([] , tf.string , default_value = ''),

      'image_name': tf.io.FixedLenFeature([] , tf.string , default_value=''),

      'patient_id': tf.io.FixedLenFeature([] , tf.int64 , default_value=0),

      'sex': tf.io.FixedLenFeature([] , tf.int64 , default_value=0),

      'age_approx': tf.io.FixedLenFeature([] , tf.int64 , default_value=0),

      'anatom_site_general_challenge':tf.io.FixedLenFeature([] ,tf.int64 , default_value=0 ),

      'diagnosis': tf.io.FixedLenFeature([] ,tf.int64 , default_value=0 ),

      'target': tf.io.FixedLenFeature([] ,tf.int64 , default_value=0 ),

      'width': tf.io.FixedLenFeature([] ,tf.int64 , default_value=0 ),

      'height': tf.io.FixedLenFeature([] ,tf.int64 , default_value=0 )

    }

    

    p = tf.io.parse_single_example(sample , features)

    

    img = p['image']

    target = p['target']

    

    return img , target
def decode_img(img , IMG_DIMS):

    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.resize(img , [IMG_DIMS , IMG_DIMS])

    img = img/255

    return img
features = {

  'image': tf.io.FixedLenFeature([] , tf.string , default_value = ''),

  'image_name': tf.io.FixedLenFeature([] , tf.string , default_value=''),

  'patient_id': tf.io.FixedLenFeature([] , tf.int64 , default_value=0),

  'sex': tf.io.FixedLenFeature([] , tf.int64 , default_value=0),

  'age_approx': tf.io.FixedLenFeature([] , tf.int64 , default_value=0),

  'anatom_site_general_challenge':tf.io.FixedLenFeature([] ,tf.int64 , default_value=0 ),

  'diagnosis': tf.io.FixedLenFeature([] ,tf.int64 , default_value=0 ),

  'target': tf.io.FixedLenFeature([] ,tf.int64 , default_value=0 ),

  'width': tf.io.FixedLenFeature([] ,tf.int64 , default_value=0 ),

  'height': tf.io.FixedLenFeature([] ,tf.int64 , default_value=0 )

}

GCS_PATH = KaggleDatasets().get_gcs_path('melanoma-256x256')

train_datasets = tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')



print(len(train_datasets))
def parsed_TFR_unlabelled(sample):

    feature_description = {

        'image': tf.io.FixedLenFeature([], tf.string, default_value=''),

        'image_name': tf.io.FixedLenFeature([], tf.string, default_value=''),

        'target': tf.io.FixedLenFeature([], tf.int64, default_value=0),

    }

    p = tf.io.parse_single_example(sample , feature_description)

    img = p['image']

    

    return img
def _get_ds(files , shuffle=True , labelled=True , repeat=True , img_dims=64 , batch_size=32):

    ds = tf.data.TFRecordDataset(files , num_parallel_reads=AUTO)

    ds = ds.cache()

    

    if repeat:

        ds = ds.repeat()

    if labelled:

        ds = ds.map(parse_TFR_data_labelled , num_parallel_calls=AUTO)

        ds = ds.map(lambda img , label:(decode_img(img , img_dims) , label) , num_parallel_calls=AUTO)

    else:

        ds = ds.map(parsed_TFR_unlabelled , num_parallel_calls=AUTO)

        ds = ds.map(lambda img : decode_img(img, img_dims) , num_parallel_calls=AUTO)

    

    ds = ds.batch(batch_size*REPLICAS)

    ds = ds.prefetch(AUTO)

    return ds

    
#lr_schedule

def lr_schedule(batch_size= 16):

    lr_start = 0.000005

    lr_max = 0.00000125 * REPLICAS * batch_size

    lr_min = 0.000001

    lr_ramp_ep = 5

    lr_sus_ep = 0

    lr_decay = 0.8

    def lrfn(epoch):

        if epoch < lr_ramp_ep:

            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start



        elif epoch < lr_ramp_ep + lr_sus_ep:

            lr = lr_max



        else:

            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min



        return lr



    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)

    return lr_callback



def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 

         for filename in filenames]

    return np.sum(n)
def create_model(input_dim , efnt=True , efnt_n=0):

    efnt_b=[efn.EfficientNetB0 , efn.EfficientNetB1 , efn.EfficientNetB2 , efn.EfficientNetB3]

    if efnt:

        base = efnt_b[efnt_n](input_shape = (input_dim , input_dim , 3), weights = 'imagenet' , include_top=False)

    else:

        base = tf.keras.applications.MobileNetV2(input_shape=(input_dim , input_dim,3),

                                               include_top=False,

                                               weights='imagenet')

    for layers in base.layers[15:]:

        layers.trainable = True

    in_put = Input(shape = (input_dim , input_dim , 3))

    x = base(in_put)

    x = GlobalAveragePooling2D()(x)

    x = Dense(64 , activation='relu')(x)

    x = Dropout(0.3)(x)

    x = Dense(32 , activation='relu')(x)

    x = Dense(1 , activation='sigmoid')(x)

    model = Model(inputs=in_put , outputs=x)

    model.compile(optimizer='adam' ,loss='binary_crossentropy' , metrics=['Accuracy' , 'AUC'])

    return model
TPU = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(TPU)

tf.tpu.experimental.initialize_tpu_system(TPU)

strategy = tf.distribute.experimental.TPUStrategy(TPU)



AUTO     = tf.data.experimental.AUTOTUNE

REPLICAS = strategy.num_replicas_in_sync
skf = StratifiedKFold(n_splits=SPLITS)

kf = KFold(n_splits=SPLITS)

oof_pred =[]

oof_train =[]

oof_val = []

oof_hist = []

f = 0

for idxT , idxV in kf.split(train_datasets):

    #print(idxT , idxV)

    train = []

    val = []

    for idx in idxT:

        train.append(train_datasets[idx])

    for idx in idxV:

        val.append(train_datasets[idx])

        

    with strategy.scope():

        lr_callback = lr_schedule(BATCH_SIZE)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = 'model_weights_fold_'+str(f)+'.hdf5' , 

                                                     save_best_only=True , verbose=1)

    

        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_auc' , verbose=1, 

                                                   patience=5 , mode='max' ,

                                                   restore_best_weights=True)



        model = create_model(IMG_DIMS)

        history = model.fit(_get_ds(train), 

                            epochs=20 ,

                            steps_per_epoch= count_data_items(train)/BATCH_SIZE//REPLICAS, 

                            validation_data = _get_ds(val ,repeat=False),

                            callbacks = [lr_callback ,cp_callback , es_callback],

                            verbose=0)



        oof_hist.append(history)

        x_val = _get_ds(val , labelled=False , repeat=False)

        oof_val.append(x_val)

        preds = model.predict(x_val)

        oof_pred.append(preds)

    

    f +=1

    
fig = plt.figure(figsize=(8,8))

i = 1

for p in oof_hist:

    fig.add_subplot(2,2,i)

    plt.plot(p.history['loss'])

    plt.plot(p.history['val_loss'])

    plt.title('fold '+str(i)+ ' LOSS')

    i +=1



plt.legend(labels= ['loss' ,'val_loss'])

fig.tight_layout(pad=3)

plt.show()
fig = plt.figure(figsize=(8,8))

i = 1

for p in oof_hist:

    fig.add_subplot(2,2,i)

    plt.plot(p.history['auc'])

    plt.plot(p.history['val_auc'])

    plt.title('fold '+str(i) + ' AUC')

    i +=1



plt.legend(labels= ['auc' ,'val_auc'])

fig.tight_layout(pad=3)

plt.show()
fig = plt.figure(figsize=(8,8))

i = 1

for p in oof_hist:

    fig.add_subplot(2,2,i)

    plt.plot(p.history['accuracy'])

    plt.plot(p.history['val_accuracy'])

    plt.title('fold '+str(i) + ' ACCURACY')

    i +=1



plt.legend(labels= ['accuracy' ,'val_accuracy'])

fig.tight_layout(pad=3)

plt.show()
# lets modify our parse function to also save the image name

def parsed_TFR_unlabelled_2(sample):

    feature_description = {

        'image': tf.io.FixedLenFeature([], tf.string, default_value=''),

        'image_name': tf.io.FixedLenFeature([], tf.string, default_value=''),

        'target': tf.io.FixedLenFeature([], tf.int64, default_value=0),

    }

    p = tf.io.parse_single_example(sample , feature_description)

    img = p['image']

    name = p['image_name']

    return name , img
GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')

train_datasets = tf.io.gfile.glob(GCS_PATH + '/tfrecords/test*.tfrec')



print(len(train_datasets))
test_data= tf.data.TFRecordDataset(train_datasets)

test_data = test_data.map(parsed_TFR_unlabelled_2 , num_parallel_calls=AUTO)

test_data = test_data.map(lambda name , img: (name , decode_img(img , 64)))



sub_df = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')

sub_df.head()
x_dict = {}

for p in test_data:

    temp = {p[0].numpy().decode() : p[1].numpy()}

    x_dict.update(temp)
len(x_dict)
test = []

for i in sub_df['image_name']:

    test.append(x_dict[i])

    del(x_dict[i])

    

test = np.array(test)

print(test.shape)
preds = model.predict(test)
np.save('test.npy' , test)
sub_df['target'] = preds

sub_df.set_index('image_name' , inplace=True)

sub_df.head()
sub_df.to_csv('submission.csv')
kf = KFold(n_splits=SPLITS)

oof_pred2 =[]

oof_train2 =[]

oof_val2 = []

oof_hist2 = []

f = 0

for idxT , idxV in kf.split(train_datasets):

    #print(idxT , idxV)

    train = []

    val = []

    for idx in idxT:

        train.append(train_datasets[idx])

    for idx in idxV:

        val.append(train_datasets[idx])

        

    with strategy.scope():

        lr_callback = lr_schedule(BATCH_SIZE)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = 'model2_weights_fold_'+str(f)+'.hdf5' , 

                                                     save_best_only=True , verbose=1)

    

        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_auc' , verbose=1, 

                                                   patience=5 , mode='max' ,

                                                   restore_best_weights=True)



        model2 = create_model(IMG_DIMS ,efnt=False)

        history = model2.fit(_get_ds(train), 

                            epochs=20 ,

                            steps_per_epoch= count_data_items(train)/BATCH_SIZE//REPLICAS, 

                            validation_data = _get_ds(val ,repeat=False),

                            callbacks = [lr_callback ,cp_callback , es_callback],

                            verbose=0)



        oof_hist2.append(history)

        x_val = _get_ds(val , labelled=False , repeat=False)

        oof_val2.append( _get_ds(val ,repeat=False))

        preds = model.predict(x_val)

        oof_pred2.append(preds)

    

    f +=1
model2.summary()
fig = plt.figure(figsize=(8,8))

i = 1

for p in oof_hist2:

    fig.add_subplot(2,2,i)

    plt.plot(p.history['loss'])

    plt.plot(p.history['val_loss'])

    plt.title('fold '+str(i)+ ' LOSS')

    i +=1



plt.legend(labels= ['loss' ,'val_loss'])

fig.tight_layout(pad=3)

plt.show()
fig = plt.figure(figsize=(8,8))

i = 1

for p in oof_hist2:

    fig.add_subplot(2,2,i)

    plt.plot(p.history['auc'])

    plt.plot(p.history['val_auc'])

    plt.title('fold '+str(i) + ' AUC')

    i +=1



plt.legend(labels= ['auc' ,'val_auc'])

fig.tight_layout(pad=3)

plt.show()
fig = plt.figure(figsize=(8,8))

i = 1

for p in oof_hist2:

    fig.add_subplot(2,2,i)

    plt.plot(p.history['accuracy'])

    plt.plot(p.history['val_accuracy'])

    plt.title('fold '+str(i) + ' ACCURACY')

    i +=1



plt.legend(labels= ['accuracy' ,'val_accuracy'])

fig.tight_layout(pad=3)

plt.show()
preds2 = model2.predict(test)
sub_df['target'] = preds2

# sub_df.set_index('image_name' , inplace=True)

sub_df.head()
sub_df.to_csv('submission2.csv')
a = tf.data.TFRecordDataset(train_datasets[:2])

a = a.map(parse_TFR_data_labelled)

a = a.map(lambda img,label:(decode_img(img ,64) , label))

fig = plt.figure(figsize=(12,10))

i = 1

for p in a.take(6):

    fig.add_subplot(2,3,i)

    img = p[0].numpy()

    l = p[1].numpy()

    plt.imshow(img)

    plt.title('Normal Image '+str(i)+'\n Label = '+str(l))

    i += 1
fig = plt.figure(figsize=(12,10))

i = 1

for p in a.take(6):

    fig.add_subplot(2,3,i)

    img = p[0]

    l = p[1].numpy()

    img = tf.image.random_brightness(img , 0.3)

    plt.imshow(img.numpy())

    plt.title('Random Brightness '+ str(i)+'\n Label = '+str(l))

    i+=1
fig = plt.figure(figsize=(12,10))

i = 1

for p in a.take(6):

    fig.add_subplot(2,3,i)

    img = p[0]

    l = p[1].numpy()

    img = tf.image.random_contrast(img , 1 ,4)

    plt.imshow(img.numpy())

    plt.title('Random Contrast ' + str(i) + '\n Label = '+str(l))

    i += 1
fig = plt.figure(figsize=(12,10))

i = 1

for p in a.take(6):

    fig.add_subplot(2,3,i)

    img = p[0]

    l = p[1].numpy()

    img = tf.image.rgb_to_grayscale(img)

    plt.imshow(img.numpy().reshape(64,64) , cmap='gray')

    plt.title('GrayScale ' + str(i)+ '\n Label = '+str(l))

    i += 1
fig = plt.figure(figsize=(12,10))

i = 1

for p in a.take(6):

    fig.add_subplot(2,3,i)

    img = p[0]

    l = p[1].numpy()

    img = tf.image.random_saturation(img , 1,3)

    plt.imshow(img.numpy())

    plt.title('flip left right ' + str(i) + '\n Label = '+str(l))

    i += 1
fig = plt.figure(figsize=(12,10))

i = 1

for p in a.take(6):

    fig.add_subplot(2,3,i)

    img = p[0]

    l = p[1].numpy()

    img = tfa.image.gaussian_filter2d(img , sigma=1)

    plt.imshow(img.numpy())

    plt.title('Gaussian Blur - RGB ' + str(i) + '\n Label = '+str(l))

    i += 1
fig = plt.figure(figsize=(12,10))

i = 1

for p in a.take(6):

    fig.add_subplot(2,3,i)

    img = p[0]

    l = p[1].numpy()

    img = tf.image.rgb_to_grayscale(img)

    img = tfa.image.gaussian_filter2d(img, sigma=1)

    plt.imshow(img.numpy().reshape(64,64) , cmap='gray')

    plt.title('Gaussian Blur - GrayScale ' + str(i)+ '\n Label = '+str(l))

    i += 1
# FLIP

fig = plt.figure(figsize=(12,10))

i = 1

for p in a.take(6):

    fig.add_subplot(2,3,i)

    img = p[0]

    l = p[1].numpy()

    img = tf.image.random_flip_left_right(img)

    plt.imshow(img.numpy())

    plt.title('flip left right ' + str(i) + '\n Label = '+str(l))

    i += 1
# Rotation Random

fig = plt.figure(figsize=(12,10))

i = 1

for p in a.take(6):

    fig.add_subplot(2,3,i)

    img = p[0]

    l = p[1].numpy()

    img = tf.image.rot90(img)

    plt.imshow(img.numpy())

    plt.title('flip left right ' + str(i) + '\n Label = '+str(l))

    i += 1
# central crop

fig = plt.figure(figsize=(12,10))

i = 1

for p in a.take(6):

    fig.add_subplot(2,3,i)

    img = p[0]

    l = p[1].numpy()

    img = tf.image.central_crop(img , 0.7)

    img = tf.image.resize(img , (64, 64))

    plt.imshow(img.numpy())

    plt.title('flip left right ' + str(i) + '\n Label = '+str(l))

    i += 1