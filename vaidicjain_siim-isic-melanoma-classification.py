!pip install -q efficientnet

import tensorflow as tf

import pandas as pd

import numpy as np

import os 

import gc

from kaggle_datasets import KaggleDatasets

from PIL import Image

from keras.metrics import *

from keras.preprocessing import image

from tensorflow.keras.layers import Input

from tensorflow.keras.applications.inception_v3 import InceptionV3

import efficientnet.tfkeras as efn

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')
train_images_dir = GCS_PATH+'/jpeg/train/'

test_images_dir = GCS_PATH+'/jpeg/test/'

train_csv = GCS_PATH+'/train.csv'

test_csv  = GCS_PATH+'/test.csv'


train_df = pd.read_csv(train_csv)

test_df = pd.read_csv(test_csv)



train_df = train_df.replace('male', 0)

train_df = train_df.replace('female', 1)



test_df = test_df.replace('male', 0)

test_df = test_df.replace('female', 1)



train_df = train_df.dropna()



train_df = train_df.drop(columns = 'patient_id')

train_df = train_df.drop(columns = 'diagnosis')

train_df = train_df.drop(columns = 'benign_malignant')

test_df = test_df.drop(columns = 'patient_id')



#anatom_site_general_challenge

train_df = train_df.replace('torso', 0)

train_df = train_df.replace('lower extremity', 1)

train_df = train_df.replace('upper extremity', 2)

train_df = train_df.replace('head/neck', 3)

train_df = train_df.replace('palms/soles', 4)

train_df = train_df.replace('oral/genital', 5)

train_df = train_df.replace(np.nan, 6)



test_df = test_df.replace('torso', 0)

test_df = test_df.replace('lower extremity', 1)

test_df = test_df.replace('upper extremity', 2)

test_df = test_df.replace('head/neck', 3)

test_df = test_df.replace('palms/soles', 4)

test_df = test_df.replace('oral/genital', 5)

test_df = test_df.replace(np.nan, 6)



# del test_csv

# del train_csv

# gc.collect()

train_df
# metadata_train = train_df[['sex','age_approx','anatom_site_general_challenge']]

# target_train = train_df[['target']]

# img_train = train_df['image_name']+train_images_dir



# metadata_test = test_df[['sex','age_approx','anatom_site_general_challenge']]

# img_test = test_df[['image_name']]



# data_metadata_train = tf.data.Dataset.from_tensor_slices(metadata_train.values)

# data_target_train = tf.data.Dataset.from_tensor_slices(target_train.values)

# data_img_train = tf.data.Dataset.from_tensor_slices(img_train.values)



# img_train = tf.convert_to_tensor(img_train.values)

#img_train = tf.io.decode_csv(img_train, record_defaults = ['string'])



image_path = train_images_dir + train_df['image_name'] + '.jpg'

image_path = tf.convert_to_tensor(image_path, dtype=tf.string)

metadata = train_df[['sex','age_approx','anatom_site_general_challenge']]

metadata = np.array(metadata)

metadata = tf.convert_to_tensor(metadata)

labels = np.array(train_df[['target']])

print(labels.sum())

print(np.shape(labels))

labels = tf.convert_to_tensor(labels)



timage_path = test_images_dir + test_df['image_name'] + '.jpg'

timage_path = tf.convert_to_tensor(timage_path, dtype=tf.string)

tmetadata = test_df[['sex','age_approx','anatom_site_general_challenge']]

tmetadata = np.array(tmetadata)

tmetadata = tf.convert_to_tensor(tmetadata)



dataset = tf.data.Dataset.from_tensor_slices((image_path, metadata, labels))



tdataset = tf.data.Dataset.from_tensor_slices((timage_path, tmetadata))



def map_fn(path,metadata,label):

    

    image = tf.image.decode_jpeg(tf.io.read_file(path))



    image = tf.image.resize(image, [128, 128])

    

    image = tf.cast(image, tf.float32) / 255.0

    

    return (image, metadata), label



def tmap_fn(path,metadata):

    

    image = tf.image.decode_jpeg(tf.io.read_file(path))



    image = tf.image.resize(image, [128, 128])

    

    image = tf.cast(image, tf.float32) / 255.0

    #test = (image, metadata)

    return (image, metadata)





# # detect and init the TPU

# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

# tf.config.experimental_connect_to_cluster(tpu)

# tf.tpu.experimental.initialize_tpu_system(tpu)



# # instantiate a distribution strategy

# strategy = tf.distribute.experimental.TPUStrategy(tpu)



batch_size = 16 * strategy.num_replicas_in_sync

dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset = dataset.repeat()

dataset = dataset.batch(batch_size)

#33126



dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)



#images, metadatas, labels = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
# train_imgs = []

# train_metadatas = []

# labels = []



# for index, row in train_df.iterrows():

#     label = row['target']

#     labels.append(label)

    

#     metadata = row[0:-1]

#     train_metadatas.append(metadata)

    

#     train_img = image.load_img(train_images_dir + index + '.jpg', target_size = (128,128,3))

#     train_img = image.img_to_array(train_img, dtype = np.float16)

#     train_img = train_img / 225.0

#     train_imgs.append(train_img)

    

    

    

# labels = np.array(labels).astype(np.float16)

# train_metadatas = np.array(train_metadatas).astype(np.float16)

# train_imgs = np.array(train_imgs)
# del train_df

# gc.collect()
with strategy.scope():

    input_tensor = Input(shape=(128, 128, 3))

    base_model = efn.EfficientNetB7(include_top=False,

        input_shape=(128,128, 3),

        weights='imagenet'

    )(input_tensor)

    base_model.trainable = True  

    

    #input_tensor = Input(shape=(128, 128, 3))

    #base_model = efn.EfficientNetB0(include_top=False,input_shape=(128, 128, 3),weights='imagenet')



    #base_model = InceptionV3(include_top=False, input_tensor=input_tensor, weights='imagenet', input_shape=(128,128,3))

    #x = base_model.output

    x = GlobalAveragePooling2D()(base_model)

    x = Dense(512, activation='relu')(x)

    predictions = Dense(16, activation='relu')(x)



#    img_model = Model(inputs=input_tensor, outputs=predictions)



    input_data = Input(shape=(3))

    metadata_model = Dense(3, activation="relu")(input_data)

#    metadata_model = Model(inputs=input_data, outputs = metadata_model)



    combined = tf.keras.layers.concatenate([predictions, metadata_model], axis = 1)



    z = Dense(4, activation="relu")(combined)

    z = Dense(1, activation="sigmoid")(z)



    model = Model(inputs=[input_tensor, input_data], outputs=z)

    

    

    METRICS = [

      BinaryAccuracy(name='accuracy'),

      AUC(name='auc'),

    ]

    model.compile(

        optimizer = 'adam',

        loss = 'binary_crossentropy',

        metrics=[METRICS]

    )
# input_tensor = Input(shape=(128, 128, 3))

# #base_model = efn.EfficientNetB0(include_top=False,input_shape=(128, 128, 3),weights='imagenet')



# #base_model = InceptionV3(include_top=False, input_tensor=input_tensor, weights='imagenet', input_shape=(128,128,3))

# x = base_model.output

# x = GlobalAveragePooling2D()(x)

# x = Dense(512, activation='relu')(x)

# predictions = Dense(16, activation='relu')(x)



# img_model = Model(inputs=base_model.input, outputs=predictions)



# input_data = Input(shape=(3))

# metadata_model = Dense(3, activation="relu")(input_data)

# metadata_model = Model(inputs=input_data, outputs = metadata_model)



# combined = tf.keras.layers.concatenate([img_model.output, metadata_model.output])



# z = Dense(4, activation="relu")(combined)

# z = Dense(1, activation="sigmoid")(z)



# model = Model(inputs=[img_model.input, metadata_model.input], outputs=z)
# METRICS = [

#       BinaryAccuracy(name='accuracy'),

#       AUC(name='auc'),

# ]

# model.compile(

#     optimizer = 'adam',

#     loss = 'binary_crossentropy',

#     metrics=[METRICS]

# )
#model.fit([images, metadatas], labels, validation_split=0.30, epochs = 5, batch_size = 1)

steps = 33126 // batch_size

weights = {0:0.018,

           1:0.982

          }

model.fit(dataset, epochs = 5, steps_per_epoch=steps, class_weight = weights)
# del images

# del metadatas

# del labels

# gc.collect()

tdataset = tdataset.map(tmap_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

tdataset = tdataset.batch(10982)

#10982

tdataset = tdataset.prefetch(tf.data.experimental.AUTOTUNE)

timages, tmetadatas = tf.compat.v1.data.make_one_shot_iterator(tdataset).get_next()
# test_imgs = []

# test_metadatas = []



# for index, row in test_df.head().iterrows():

#     metadata = row[0:]

#     test_metadatas.append(metadata)

    

#     test_img = image.load_img(test_images_dir + index + '.jpg', target_size = (128,128,3))

#     test_img = image.img_to_array(test_img, dtype = np.float16)

#     test_img = test_img / 225.0

#     test_imgs.append(test_img)

    



# test_metadatas = np.array(test_metadatas).astype(np.float16)

# test_imgs = np.array(test_imgs)
test_labels = model.predict([timages, tmetadatas])

np.shape(test_labels)

test_df = test_df.reset_index()

df_sub = pd.DataFrame()

df_sub['image_name'] = test_df['image_name']

df_sub['target'] = test_labels.astype(np.float32)

df_sub.to_csv('submission.csv', index=False)