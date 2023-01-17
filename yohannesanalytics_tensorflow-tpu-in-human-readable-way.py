!pip install -q efficientnet
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers,models

import matplotlib.pyplot as plt

from functools import partial
from sklearn.model_selection import train_test_split

from kaggle_datasets import KaggleDatasets

import efficientnet.tfkeras as efn
GCS_PATH = KaggleDatasets().get_gcs_path()
TRAINING_FILENAMES, VALIDATION_FILENAMES = train_test_split(
    tf.io.gfile.glob(GCS_PATH + '/tfrecords/train*.tfrec'),
    test_size=0.2, random_state=11
)
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/test*.tfrec')
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)
AUTOTUNE=tf.data.experimental.AUTOTUNE
BATCH_SIZE_TRAIN=16*strategy.num_replicas_in_sync
BATCH_SIZE_TEST=1024
EPOCHS=20
def view_image(batch,batch_size):
    
    try:
        image,label=next(iter(batch))
        image=image.numpy()
        label=label.numpy()
    except:
        image=next(iter(batch))
        image=image.numpy()
    
    fig=plt.figure(figsize=(22,22))
    
    for i in range(batch_size):
            ax=fig.add_subplot(4,8,i+1)
            ax.imshow(image[i])
            try:
                ax.set_title(f"Label: {label[i]}")
            except:
                pass

def augmentations(image,label,augment):
    
    if augment:
        image=tf.image.random_flip_left_right(image)
        image=tf.image.random_flip_up_down(image)
        image=tf.image.random_hue(image,0.3)
        image=tf.image.random_contrast(image,0.2,0.5)
        image=tf.image.random_brightness(image,0.3)
        image=tf.image.random_saturation(image,5,10)
        
    image=tf.cast(image,tf.float32)/255.0
    image=tf.image.resize(image,[256,256])
    return image,label

def standart_parse(image):
    image=tf.cast(image,tf.float32)/255.0
    image=tf.image.resize(image,[256,256])
    return image

def decode_image(image):
    image=tf.image.decode_jpeg(image,channels=3)
    return image

def read_tfrecord(record, labeled):
    # scheme for tfrecord parsing
    
    if labeled:
        TFRECORD_SCHEME={

            "image": tf.io.FixedLenFeature([],tf.string),
            "target": tf.io.FixedLenFeature([],tf.int64)
        }
    else:
        TFRECORD_SCHEME={
            "image": tf.io.FixedLenFeature([],tf.string),
            "image_name": tf.io.FixedLenFeature([],tf.string)
        }        
    
    datapoint = tf.io.parse_single_example(record,TFRECORD_SCHEME)
    image = decode_image(datapoint["image"])
    if labeled:
        label = datapoint["target"]
        return image,label
    else:
        return image

def load_dataset(filenames,labeled,batch_size,augment=True):
    
    options=tf.data.Options()
    options.experimental_deterministic=False
    
    dataset=tf.data.TFRecordDataset(filenames,num_parallel_reads=AUTOTUNE)
    
    dataset=dataset.with_options(options)
    
    dataset=dataset.map(partial(read_tfrecord,labeled=labeled),num_parallel_calls=AUTOTUNE)
    
    if labeled:
        dataset=dataset.map(partial(augmentations,augment=augment),num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    else:
        dataset=dataset.map(standart_parse,num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
        
    return dataset
    
melanoma_train=load_dataset(TRAINING_FILENAMES,labeled=True,batch_size=BATCH_SIZE_TRAIN)
melanoma_val=load_dataset(VALIDATION_FILENAMES,labeled=True,augment=False,batch_size=BATCH_SIZE_TEST)
melanoma_test=load_dataset(TEST_FILENAMES,labeled=False,augment=False,batch_size=BATCH_SIZE_TEST)

with strategy.scope():
    B7efn=efn.EfficientNetB7(weights="noisy-student",include_top=False,input_shape=[256,256,3])
    model = models.Sequential([
        B7efn,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.1),
        layers.Dense(128,activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(64,activation="relu"),
        layers.Dense(1,activation="sigmoid")
    ])
    

# testing=models.Sequential([


#         layers.Conv2D(32, (3, 3), activation='relu', input_shape=[256,256,3]),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(64, (3, 3), activation='relu'),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(64, (3, 3), activation='relu'),
#         layers.Flatten(),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(1, activation='sigmoid')])

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics="accuracy")
model_checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath="best_model.h5",save_weights_only=True,save_best_only=True,monitor="val_accuracy",mode="max")
lrdecay=tf.keras.callbacks.ReduceLROnPlateau(patience=3,verbose=1)

model.fit(melanoma_train,validation_data=melanoma_val, epochs=EPOCHS, callbacks=[model_checkpoint,lrdecay])
