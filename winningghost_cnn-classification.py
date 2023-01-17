import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt
train_folder = '/kaggle/input/intel-image-classification/seg_train/seg_train/'
import os
#find number of files in train folder

n = 0

for _, _, filenames in os.walk(train_folder):

    for filename in filenames:

        n += 1
n
ds_filelist_train = tf.data.Dataset.list_files(file_pattern = train_folder + '*/*', shuffle = False)
ds_filelist_train = ds_filelist_train.shuffle(buffer_size = n, reshuffle_each_iteration = False)
table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(tf.constant(['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']), tf.range(6)), -1)

def load_imgs(filepath):

    img = tf.io.read_file(filepath)

    img = tf.io.decode_jpeg(img, channels = 3)

    img = tf.image.convert_image_dtype(img, tf.float32)

    img = tf.image.resize(img, [150, 150])

    

    tag = tf.strings.split(filepath, os.path.sep)[-2]



    label = table.lookup(tag)

    

    return img, label
ds_train = ds_filelist_train.map(load_imgs, num_parallel_calls = tf.data.experimental.AUTOTUNE)
def img_augmentation(img, label):

    #img = tf.image.random_contrast(img, 0.1, 0.9)

    #img = tf.image.random_hue(img, 0.5)

    img = tf.image.random_flip_left_right(img)

    img = tf.image.random_crop(img, (125, 125, 3))

    return tf.image.resize(img, (150, 150)), label
ds_train = ds_train.map(img_augmentation, num_parallel_calls = tf.data.experimental.AUTOTUNE)
for i in ds_train.take(1):

    print(i)
#Tranfer Learning using InceptionV3

Inception_v3 = tf.keras.applications.InceptionV3(include_top = False, input_shape = (150, 150, 3), weights = None)



#load weights

local_weights_file = '/kaggle/input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

Inception_v3.load_weights(local_weights_file)
Inception_v3.summary()
#freeze layers

for layer in Inception_v3.layers:

    layer.trainable = False
#get last layer

last_layer = Inception_v3.layers[-1]

print(last_layer.output_shape)

last_output = last_layer.output
x = tf.keras.layers.Flatten()(last_output)

x = tf.keras.layers.Dense(1024, activation = 'relu')(x)

x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Dense(6)(x)



model = tf.keras.Model(inputs = Inception_v3.input, outputs = x)
model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])
#Prepare the data for training

ds_train_orig = ds_train

ds_train = ds_train_orig.take(13000)

ds_valid = ds_train_orig.skip(13000)



#shuffle, batch

ds_train = ds_train.shuffle(n).batch(20).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

ds_valid = ds_valid.batch(20).cache().prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
#Adding Callbacks

callback_1 = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 3, verbose = 1)

callback_2 = tf.keras.callbacks.ModelCheckpoint(filepath = '/kaggle/working/model_checkpoints/checkpoint_{epoch}.ckpt', save_weights_only = True, verbose = 1)
history = model.fit(ds_train, epochs = 20, validation_data = ds_valid, callbacks = [callback_1, callback_2])
hist = history.history
epochs = np.arange(len(hist['loss'])) + 1



fig, ax = plt.subplots(1, 2, figsize = (20, 8))



ax[0].plot(epochs, hist['loss'], '-o', label = 'training_loss')

ax[0].plot(epochs, hist['val_loss'], '--<', label = 'val_loss')

ax[0].legend()

ax[0].set_title('Training vs. Val Loss')



ax[1].plot(epochs, hist['accuracy'], '-o', label = 'training_accuracy')

ax[1].plot(epochs, hist['val_accuracy'], '--<', label = 'val_accuracy')

ax[1].legend()

ax[1].set_title('Training vs. Val Accuracy')

test_folder = '/kaggle/input/intel-image-classification/seg_test/seg_test/'

ds_filelist_test = tf.data.Dataset.list_files(file_pattern = test_folder + '*/*', shuffle = False)

ds_test = ds_filelist_test.map(load_imgs, num_parallel_calls = tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(20)
model.evaluate(ds_test)
ds_sample = ds_test.unbatch().shuffle(n, reshuffle_each_iteration = False).batch(10).take(1)
pred = model.predict(ds_sample)
pred_prob = tf.nn.softmax(pred, axis = 1)

pred_label = tf.math.argmax(pred_prob, axis = 1)

pred_label_prob = tf.math.reduce_max(pred_prob, axis = 1)
rev_table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(tf.range(6), tf.constant(['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'])), 'x')

pred_tag = rev_table.lookup(tf.cast(pred_label, tf.int32))
fig = plt.figure(figsize = (15, 8))

for i, sample in enumerate(ds_sample.unbatch()):

    img = sample[0]

    ax = fig.add_subplot(2, 5, i + 1)

    ax.imshow(img)

    ax.set_xticks([])

    ax.set_yticks([])

    

    actual_tag = rev_table.lookup(sample[1])

    ax.text(0.5, -0.15, f'Actual: {actual_tag.numpy()}\nPredicted: {pred_tag[i].numpy()}\nProbability: {pred_label_prob[i].numpy():.2%}',

            size = 14, horizontalalignment = 'center', verticalalignment = 'center', transform = ax.transAxes)

    

plt.tight_layout()

plt.show()