# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

from kaggle_datasets import KaggleDatasets

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow_addons as tfa

from tensorflow.keras.layers import Dense , Dropout , BatchNormalization , concatenate

from tensorflow.keras.layers import Activation , Input , GlobalAveragePooling2D  

from tensorflow.keras.models import Model , Sequential , load_model

from tensorflow.keras.applications import InceptionV3 , MobileNetV2

from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression

import re

from PIL import Image
IMG_DIMS = 64 # inception net has a minimum input size of 75x75 thus 150 is good

CHANNELS = 3

BATCH_SIZE = 32

SEED = 42

SPLITS = 5



AUTO  = tf.data.experimental.AUTOTUNE
# getting training data gcs path

GCS_PATH = KaggleDatasets().get_gcs_path('melanoma-256x256')

train_datasets = tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')



print('number of TFRecords in train : ',len(train_datasets))



# getting testing data

GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')

test_datasets = tf.io.gfile.glob(GCS_PATH + '/tfrecords/test*.tfrec')



print('number of TFRecords in test : ' ,len(test_datasets))
# parse data from TF Records

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
# decode img

def decode_image(img , IMG_DIMS):

    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.resize(img , [IMG_DIMS , IMG_DIMS])

    return img
# load data set for training and validation

def _get_ds(files , train=True , repeat=True , img_dims=64 , batch_size=32):

    ds = tf.data.TFRecordDataset(files , num_parallel_reads=AUTO)

    ds = ds.cache()

    

    if repeat:

        ds = ds.repeat()

    ds = ds.map(parse_TFR_data_labelled , num_parallel_calls=AUTO)     

    ds = ds.map(lambda img ,label: (decode_image(img, img_dims),label) , num_parallel_calls=AUTO)

    if train:

        ds = ds.shuffle(buffer_size=1000)

       

    ds = ds.batch(batch_size*REPLICAS)

    ds = ds.prefetch(AUTO)

    return ds
def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 

         for filename in filenames]

    return np.sum(n)
# preparing the testing data

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



test_data= tf.data.TFRecordDataset(test_datasets)

test_data = test_data.map(parsed_TFR_unlabelled_2 , num_parallel_calls=AUTO)

test_data = test_data.map(lambda name , img: (name , decode_image(img , 64)))



sub_df = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')



x_dict = {}

for p in test_data:

    temp = {p[0].numpy().decode() : p[1].numpy()}

    x_dict.update(temp)

    

print(f'number of samples in testing data : {len(x_dict)}')



test = []

for i in sub_df['image_name']:

    test.append(x_dict[i])

    del(x_dict[i])

    

test = np.array(test)

print(f'sahpe of testing set sorted according to the submission file : {test.shape}')
def create_Inf_model(IMG_DIMS , CHANNELS):

    in_put = Input(shape=(IMG_DIMS , IMG_DIMS , 3))

    # applying auggumentations

    #pre = aug(in_put)

    # pre process layer

    pre_process_layer = tf.keras.applications.mobilenet_v2.preprocess_input

    pre = pre_process_layer(in_put)

    # base model non trainable

    base_model = MobileNetV2(input_shape=(IMG_DIMS , IMG_DIMS , 3) , include_top=False , weights='imagenet')

    x = base_model(pre , training = False)

    # top trainable model layers

    x = GlobalAveragePooling2D()(x)

    x = Dense(128 , activation = 'relu')(x)

    x = Dropout(0.3)(x)

    x = Dense(1 , activation = 'sigmoid')(x)

    model = Model(inputs=in_put , outputs=x)

    # optimizer

    opt = tf.keras.optimizers.Adam(0.0001)

    model.compile(optimizer=opt , loss='binary_crossentropy' , metrics=['accuracy' , 'AUC'])

    return model
TPU = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(TPU)

tf.tpu.experimental.initialize_tpu_system(TPU)

strategy = tf.distribute.experimental.TPUStrategy(TPU)



REPLICAS = strategy.num_replicas_in_sync
kf = KFold(n_splits=SPLITS)

oof_hist = []

oof_val = []

for f , (idxT , idxV) in enumerate(kf.split(train_datasets)):

    train = []

    val =[]

    for idx in idxT:

        train.append(train_datasets[idx])

    for idx in idxV:

        val.append(train_datasets[idx])



    # instantiate model

    with strategy.scope():

         # cretae model check points

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='model_fold_'+str(f)+'_weights.hdf5' , 

                                                         monitor='val_auc',

                                                         mode='max',

                                                         save_best_only =True,

                                                         verbose = 1 )

        # early stopping

        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_auc' , patience=5 , mode='max' )

        model = create_Inf_model(IMG_DIMS , CHANNELS)

        

        history = model.fit(_get_ds(train) ,

                        epochs = 20 ,

                        steps_per_epoch = count_data_items(train)/BATCH_SIZE//REPLICAS,

                        validation_data=_get_ds(val , train=False , repeat=False),

                        callbacks = [cp_callback , es_callback],

                        verbose = 0)

        model.save('model_'+str(f)+'_.hdf5')

    oof_hist.append(history)

    oof_val.append(_get_ds(val, train=False))


train_data= tf.data.TFRecordDataset(train_datasets)

train_data = train_data.map(parse_TFR_data_labelled , num_parallel_calls=AUTO)

train_data = train_data.map(lambda img , target: (decode_image(img , 64) , target))



img = []

label = []

for sample in train_data:

    i = sample[0].numpy()

    t = sample[1].numpy()

    img.append(i)

    label.append(t)

    

img = np.array(img)

label = np.array(label)



print(f'shape of images : {img.shape}')

print(f'shape of labels : {label.shape}')
file_names = ['model_'+str(f)+'_.hdf5' for f in range(5)]

member_models = [load_model(m) for m in file_names]

member_models
# level0_predict function

def level0_predict(m_models , train_x):

    predictions = []

    for model in m_models:

        p = model.predict(train_x)

        predictions.append(p)

    return predictions
def stacked_set(predictions):

    X = None

    for p in predictions:

        if X is None:

            X = p

        else:

            X = np.dstack((X , p))

    X = X.reshape(X.shape[0] , X.shape[1]*X.shape[2])

    return X
def fit_agg(X , Y):

    model = Sequential([

        Dense(3 , activation='relu'),

        Dense(1 , activation ='sigmoid')

    ])

    model.compile(optimizer='adam' ,loss='mean_squared_error' , metrics=['accuracy'])

    model.fit(X,Y , epochs = 10)

    return model
def ensemble_fit(X_train , Y , X_test , member_models):

    # get prediction of each sub model

    preds = level0_predict(member_models , X_train)

    # prepare the stacked set

    X = stacked_set(preds)

    # fit aggregator

    model = fit_agg(X,Y)

    return model

    
def ensemble_predict(agg, member_models , data ):

    # get the sub model predictions

    preds = level0_predict(member_models , data)

    # prepare the stacked set

    X = stacked_set(preds)

    # final prediction

    result = agg.predict(X)

    return result
final_model = ensemble_fit(img , label , test , member_models)
result = ensemble_predict(final_model , member_models , test)
result
sub_df['target'] = result

sub_df.set_index('image_name' , inplace=True)

sub_df.head()
sub_df.to_csv('submission.csv')