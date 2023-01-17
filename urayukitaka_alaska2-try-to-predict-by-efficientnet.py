! pip install -q efficientnet
# Basic library

import numpy as np 

import pandas as pd 

import os



# Data preprocessing

from sklearn.model_selection import train_test_split



# Visualization

from matplotlib import pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')



# tensorflow

import tensorflow as tf

import tensorflow.keras.layers as l

import efficientnet.tfkeras as efn



# data set

from kaggle_datasets import KaggleDatasets
# TPU setting

# detect and init the TPU

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)



# instantiate a distribution strategy

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
print(tpu.master())

print(tpu_strategy.num_replicas_in_sync)
# For tensorflow dataset

AUTO = tf.data.experimental.AUTOTUNE

ignore_order = tf.data.Options()

ignore_order.experimental_deterministic = False



# Pass

gcs_path = KaggleDatasets().get_gcs_path()
# Sample dataframe

sample = pd.read_csv("/kaggle/input/alaska2-image-steganalysis/sample_submission.csv")
# batch size in tpu

BATCH_SIZE = 32 * tpu_strategy.num_replicas_in_sync
# Directrory and file name

# Drop csv from dir_name

dir_name = ['Test', 'JUNIWARD', 'JMiPOD', 'Cover', 'UERD']

df = pd.DataFrame({})



# Create empty dataframe and list

lists = []

cate = []



# get the filenames

for dir_ in dir_name:

    # file name

    list_ = os.listdir("/kaggle/input/alaska2-image-steganalysis/"+dir_+"/")

    lists = lists+list_

    # category name

    cate_ = np.tile(dir_,len(list_))

    cate = np.concatenate([cate,cate_])

    

# insert dataframe

df["cate"] = cate

df["name"] = lists
# path line

df["path"] = [str(os.path.join(gcs_path,cate,name)) for cate, name in zip(df["cate"], df["name"])]
# Labeling positive and negative

def cate_label(x):

    if x["cate"] == "Cover":

        res = 0

    else:

        res = 1

    return res



# Test dataframe and Train dataframe

Test_df = df.query("cate=='Test'").sort_values(by="name")

Train_df = df.query("cate!='Test'")

# Apply the function

Train_df["flg"] = df.apply(cate_label, axis=1)
# label_counts

Train_df["cate"].value_counts()
Train_df = Train_df.sample(80000)

# label_counts

Train_df["cate"].value_counts()
# Create train data and val data

X = Train_df["path"]

y = Train_df["flg"]



# split train and val data

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=10)
# Change to numpy array

X_train, X_val, y_train, y_val = np.array(X_train), np.array(X_val), np.array(y_train), np.array(y_val)
# Create test data

X_test = np.array(Test_df["path"])
def decode_image(filename, label=None, image_size=(512,512)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    image = tf.cast(image, tf.float32)/255.0

    image = tf.image.resize(image, image_size)

    

    if label is None:

        return image

    else:

        return image, label
AUTO = tf.data.experimental.AUTOTUNE

ignore_order = tf.data.Options()

ignore_order.experimental_deterministic = False



# Build input pipeline

train_dataset = (tf.data.Dataset.from_tensor_slices((X_train, y_train)).prefetch(AUTO).with_options(ignore_order)

                 .map(decode_image, num_parallel_calls=AUTO).shuffle(512).batch(BATCH_SIZE).repeat())



valid_dataset = (tf.data.Dataset.from_tensor_slices((X_val, y_val)).map(decode_image, num_parallel_calls=AUTO)

                    .cache().batch(BATCH_SIZE).prefetch(AUTO))



test_dataset = (tf.data.Dataset.from_tensor_slices((X_test)).map(decode_image, num_parallel_calls=AUTO)

                    .batch(BATCH_SIZE))
with tpu_strategy.scope():

    model_b7 = tf.keras.Sequential([

        efn.EfficientNetB7(input_shape=(512,512,3),weights='imagenet',include_top=False),

        l.GlobalAveragePooling2D(),

        l.Dense(1, activation="sigmoid")

    ])

    

    model_b7.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model_b7.summary()
STEPS_PER_EPOCH = X_train.shape[0] // BATCH_SIZE

callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]



EPOCHS = 5

hist_b7 = model_b7.fit(train_dataset, epochs=EPOCHS,

                   steps_per_epoch=STEPS_PER_EPOCH, validation_data=valid_dataset, callbacks=callbacks, workers=4, use_multiprocessing=True)
# Prediction

pred_b7 = model_b7.predict(test_dataset, verbose=1)
# training history

train_loss = hist_b7.history["loss"]

val_loss = hist_b7.history["val_loss"]

train_acc = hist_b7.history["accuracy"]

val_acc = hist_b7.history["val_accuracy"]



fig, ax = plt.subplots(1,2,figsize=(10,6))

ax[0].plot(range(len(train_loss)), train_loss, label="train_loss")

ax[0].plot(range(len(val_loss)), val_loss, label="val_loss")

ax[0].set_xlabel("epochs")

ax[0].set_ylabel("loss")

ax[0].set_title("EfficientNetB7 loss")

ax[0].legend()



ax[1].plot(range(len(train_acc)), train_acc, label="train_accuracy")

ax[1].plot(range(len(val_acc)), val_acc, label="val_accuracy")

ax[1].set_xlabel("epochs")

ax[1].set_ylabel("accuracy")

ax[1].set_title("EfficientNetB7 accurary")

ax[1].legend()
# EfficientNetB7

sample_7 = sample.copy()

sample_7["Label"] = pred_b7

sample_7.to_csv("submission_B7.csv", index=False)
sample_7["Label"].describe()