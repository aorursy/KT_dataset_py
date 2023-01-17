!pip install -q efficientnet

import pandas as pd

import numpy as np

import tensorflow as tf

import efficientnet.tfkeras as efn

from keras.metrics import *

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input

from kaggle_datasets import KaggleDatasets

from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
# GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')

# train_images_dir = GCS_PATH+'/jpeg/train/'

# test_images_dir = GCS_PATH+'/jpeg/test/'

# train_csv = GCS_PATH+'/train.csv'

# test_csv  = GCS_PATH+'/test.csv'

PATH = '/kaggle/input/siim-isic-melanoma-classification'

train_images_dir = PATH+'/jpeg/train/'

test_images_dir = PATH+'/jpeg/test/'

train_csv = PATH+'/train.csv'

test_csv  = PATH+'/test.csv'
train_df = pd.read_csv(train_csv)

test_df = pd.read_csv(test_csv)



# print(train_df.isnull().sum(axis=0))

# print(test_df.isnull().sum(axis=0))



train_df['image_name'] = train_df['image_name'] + '.jpg'

test_df['image_name'] = test_df['image_name'] + '.jpg'



train_df = train_df.sort_values(['patient_id'])

train_df = train_df.reset_index()



train_df = train_df.sort_values(['target'])

train_df = train_df.reset_index()



train_df = train_df.drop(['index', 'patient_id', 'diagnosis', 'benign_malignant'], axis = 1)

test_df = test_df.drop(['patient_id'], axis = 1)



train_df['anatom_site_general_challenge'] = train_df['anatom_site_general_challenge'].replace(np.nan, 'torso')

test_df['anatom_site_general_challenge'] = test_df['anatom_site_general_challenge'].replace(np.nan, 'torso')



train_df = train_df.dropna()



test_df['age_approx'] = test_df['age_approx'] / train_df['age_approx'].mean()

train_df['age_approx'] = train_df['age_approx'] / train_df['age_approx'].mean()



train_df['sex'] = train_df['sex'].replace('female', 0)

test_df['sex'] = test_df['sex'].replace('female', 0)

train_df['sex'] = train_df['sex'].replace('male', 1)

test_df['sex'] = test_df['sex'].replace('male', 1)



train_df['target'] = train_df['target'].replace(0,'0')

train_df['target'] = train_df['target'].replace(1,'1')



train_df['anatom_site_general_challenge'] = pd.Categorical(train_df['anatom_site_general_challenge'])

train_df['anatom_site_general_challenge'] = train_df.anatom_site_general_challenge.cat.codes

test_df['anatom_site_general_challenge'] = pd.Categorical(test_df['anatom_site_general_challenge'])

test_df['anatom_site_general_challenge'] = test_df.anatom_site_general_challenge.cat.codes



#print(train_df.isnull().sum(axis=0))

# print(test_df.isnull().sum(axis=0))





print(train_df)

test_df
val_split = 0.1

# batch_size = 16 * strategy.num_replicas_in_sync
# val = train_df[0:int(val_split * 33058)]

# train = train_df[int(val_split * 33058):]

val = train_df[24000:25000]

val = pd.concat([val, train_df[33057:]])

train = train_df[25000:33058]

# train_df['target'].sum()
# val
target_size=(128, 128)

batch_size=32
train_datagen = ImageDataGenerator(rescale=1./255.,

                                   width_shift_range=0.15,

                                   rotation_range=0.15,

                                   height_shift_range=0.15,

                                   zoom_range=0.15,

                                   horizontal_flip=True,

                                   brightness_range=[0.5,1.5]

                                  )

val_datagen = ImageDataGenerator(rescale=1./255.)



train_generator = train_datagen.flow_from_dataframe(dataframe=train,

                                                    directory=train_images_dir,

                                                    x_col='image_name',

                                                    y_col='target',

                                                    class_mode='binary',

#                                                     color_mode='grayscale',

                                                    target_size=target_size,

#                                                     shuffle=False,

                                                    batch_size=batch_size

                                                   )

val_generator = val_datagen.flow_from_dataframe(dataframe=val,

                                                directory=train_images_dir,

                                                x_col='image_name',

                                                y_col='target',

                                                class_mode='binary',

                                                target_size=target_size,

#                                                 shuffle=False,

                                                batch_size=batch_size

                                               )
with strategy.scope():

    base_model = efn.EfficientNetB0(include_top=False,

        input_shape=(128,128, 3),

        weights='imagenet'

    )

    

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(512, activation='relu')(x)

    x = Dense(128, activation='relu')(x)

    x = Dense(1, activation='sigmoid')(x)





    model = Model(inputs=base_model.input, outputs=x)

    

#     for layer in base_model.layers:

#           layer.trainable = False

    

    

    METRICS = [

      BinaryAccuracy(name='accuracy'),

      AUC(name='auc'),

    ]

    model.compile(

        optimizer = 'adam',

        loss = 'binary_crossentropy',

        metrics=[METRICS]

    )



# model = tf.keras.Sequential([efn.EfficientNetB0(include_top=False, input_shape=(128,128, 3), weights='imagenet'),

#                             GlobalAveragePooling2D(),

#                             Dense(1024, activation='relu'),

#                             Dense(512, activation='relu'),

#                             Dense(1, activation='sigmoid')]

#                            )

# METRICS = [BinaryAccuracy(name='accuracy'),AUC(name='auc'),]

# model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=[METRICS])
# weights = {0:0.018,

#            1:0.982

#           }

# weights = {0:0.072,

#            1:0.928

#           }

model.fit(x=train_generator, epochs = 7, validation_data=val_generator)#, class_weight = weights)
timage_path = test_images_dir + test_df['image_name']

timage_path = tf.convert_to_tensor(timage_path, dtype=tf.string)

tmetadata = test_df[['sex','age_approx','anatom_site_general_challenge']]

tmetadata = np.array(tmetadata)

tmetadata = tf.convert_to_tensor(tmetadata)



np.shape(timage_path)
def tmap_fn(path,metadata):

    

    image = tf.image.decode_jpeg(tf.io.read_file(path))



    image = tf.image.resize(image, [128, 128])

    

    image = tf.cast(image, tf.float32) / 255.0

    #test = (image, metadata)

    return (image, metadata)



tdataset = tf.data.Dataset.from_tensor_slices((timage_path, tmetadata))

tdataset = tdataset.map(tmap_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

tdataset = tdataset.batch(10982)

#10982

tdataset = tdataset.prefetch(tf.data.experimental.AUTOTUNE)

timages, tmetadatas = tf.compat.v1.data.make_one_shot_iterator(tdataset).get_next()



test_labels = model.predict([timages])

test_df = pd.read_csv(test_csv)



np.shape(test_labels)

test_df = test_df.reset_index()

df_sub = pd.DataFrame()

df_sub['image_name'] = test_df['image_name']

df_sub['target'] = test_labels.astype(np.float32)

df_sub.to_csv('submission.csv', index=False)