import os

from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
!ls -lha ../input/satellite-images-of-hurricane-damage
input_path = '../input/satellite-images-of-hurricane-damage/'



def print_file_sizes(input_path, subset):

    print('{}:'.format(subset))

    print('')

    path = input_path + subset + '/'

    for f in os.listdir(path):

        if not os.path.isdir(path + f):

            print(f.ljust(30) + str(round(os.path.getsize(path + f) / 1000000, 2)) + 'MB')

        else:

            sizes = [os.path.getsize(path+f+'/'+x)/1000000 for x in os.listdir(path + f)]

            print(f.ljust(30) + str(round(sum(sizes), 2)) + 'MB' + ' ({} files)'.format(len(sizes)))

    print('')

    

print_file_sizes(input_path, 'train_another')

print_file_sizes(input_path, 'validation_another')

print_file_sizes(input_path, 'test_another')

print_file_sizes(input_path, 'test')
image_df = pd.DataFrame({'path': list(Path(input_path).glob('**/*.jp*g'))})



image_df['damage'] = image_df['path'].map(lambda x: x.parent.stem)

image_df['data_split'] = image_df['path'].map(lambda x: x.parent.parent.stem)

image_df['location'] = image_df['path'].map(lambda x: x.stem)

image_df['lon'] = image_df['location'].map(lambda x: float(x.split('_')[0]))

image_df['lat'] = image_df['location'].map(lambda x: float(x.split('_')[-1]))

image_df['path'] = image_df['path'].map(lambda x: str(x)) # convert the path back to a string



image_df.head()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,6))



s = 10

alpha = 0.5



# get the train-validation-test splits

image_df_train = image_df[image_df['data_split']=='train_another'].copy()

image_df_val = image_df[image_df['data_split']=='validation_another'].copy()

image_df_test = image_df[image_df['data_split']=='test_another'].copy()



# sort to ensure reproducible behaviour

image_df_train.sort_values('lat', inplace=True)

image_df_val.sort_values('lat', inplace=True)

image_df_test.sort_values('lat', inplace=True)

image_df_train.reset_index(drop=True,inplace=True)

image_df_val.reset_index(drop=True,inplace=True)

image_df_test.reset_index(drop=True,inplace=True)



ax[0].scatter(image_df_train['lon'], image_df_train['lat'], color='C0', s=s, alpha=alpha, label='train')

ax[0].scatter(image_df_val['lon'], image_df_val['lat'], color='C1', s=s, alpha=alpha, label='validation')



ax[0].set_title('split')

ax[0].legend()

ax[0].set_xlabel('longitude')

ax[0].set_ylabel('latitude')



image_df_dmg = image_df[image_df['damage']=='damage'].copy()

image_df_nodmg = image_df[image_df['damage']=='no_damage'].copy()



image_df_dmg.reset_index(drop=True,inplace=True)

image_df_nodmg.reset_index(drop=True,inplace=True)



ax[1].scatter(image_df_dmg['lon'], image_df_dmg['lat'], color='C0', s=s, alpha=alpha, label='damage')

ax[1].scatter(image_df_nodmg['lon'], image_df_nodmg['lat'], color='C1', s=s, alpha=alpha, label='no damage')



ax[1].set_title('label')

ax[1].legend()

ax[1].set_xlabel('longitude')

ax[1].set_ylabel('latitude')



plt.show(fig)
import cv2



# read it in unchanged, to make sure we aren't losing any information

img = cv2.imread(image_df['path'][0], cv2.IMREAD_UNCHANGED)

np.shape(img)
type(img[0,0,0])
np.min(img[:,:,:])
np.max(img[:,:,:])
fig, ax = plt.subplots(nrows=4, ncols=10, sharex=True, sharey=True, figsize=(20,10))



ax = ax.flatten()



for i in range(20):

    img = cv2.imread(image_df_dmg['path'][i], cv2.IMREAD_UNCHANGED)

    ax[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    ax[i].set_title('damage')

    

for i in range(20,40):

    img = cv2.imread(image_df_nodmg['path'][i], cv2.IMREAD_UNCHANGED)

    ax[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    ax[i].set_title('no damage')

    

plt.show()
jpg_channels = ['blue','green','red']

jpg_channel_colors = ['b','g','r']



fig, ax = plt.subplots(figsize=(15,6))



for i in range(len(jpg_channels)):

    ax.hist(img[:,:,i].flatten(), bins=np.arange(256),

            label=jpg_channels[i], color=jpg_channel_colors[i], alpha=0.5)

    ax.legend()

    

ax.set_xlim(0,255)

    

plt.show(fig)
import tensorflow as tf

tf.__version__
# paths

train_path = image_df_train['path'].copy().values

val_path = image_df_val['path'].copy().values

test_path = image_df_test['path'].copy().values



# labels

train_labels = np.zeros(len(image_df_train), dtype=np.int8)

train_labels[image_df_train['damage'].values=='damage'] = 1



val_labels = np.zeros(len(image_df_val), dtype=np.int8)

val_labels[image_df_val['damage'].values=='damage'] = 1



test_labels = np.zeros(len(image_df_test), dtype=np.int8)

test_labels[image_df_test['damage'].values=='damage'] = 1
train_ds = tf.data.Dataset.from_tensor_slices((train_path, train_labels))

val_ds = tf.data.Dataset.from_tensor_slices((val_path, val_labels))

test_ds = tf.data.Dataset.from_tensor_slices((test_path, test_labels))



# note that the `numpy()` function is required to grab the actual values from the Dataset

for path, label in train_ds.take(5):

    print("path  : ", path.numpy().decode('utf-8'))

    print("label : ", label.numpy())
# this function wraps `cv2.imread` - we treat it as a 'standalone' function, and therefore can use

# eager execution (i.e. the use of `numpy()`) to get a string of the path.

# note that no tensorflow functions are used here

def cv2_imread(path, label):

    # read in the image, getting the string of the path via eager execution

    img = cv2.imread(path.numpy().decode('utf-8'), cv2.IMREAD_UNCHANGED)

    # change from BGR to RGB

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img, label



# this function assumes that the image has been read in, and does some transformations on it

# note that only tensorflow functions are used here

def tf_cleanup(img, label):

    # convert to Tensor

    img = tf.convert_to_tensor(img)

    # unclear why, but the jpeg is read in as uint16 - convert to uint8

    img = tf.dtypes.cast(img, tf.uint8)

    # set the shape of the Tensor

    img.set_shape((128, 128, 3))

    # convert to float32, scaling from uint8 (0-255) to float32 (0-1)

    img = tf.image.convert_image_dtype(img, tf.float32)

    # resize the image

    img = tf.image.resize(img, [128, 128])

    # convert the labels into a Tensor and set the shape

    label = tf.convert_to_tensor(label)

    label.set_shape(())

    return img, label



AUTOTUNE = tf.data.experimental.AUTOTUNE



# map the cv2 wrapper function using `tf.py_function`

train_ds = train_ds.map(lambda path, label: tuple(tf.py_function(cv2_imread, [path, label], [tf.uint16, label.dtype])),

                        num_parallel_calls=AUTOTUNE)

val_ds = val_ds.map(lambda path, label: tuple(tf.py_function(cv2_imread, [path, label], [tf.uint16, label.dtype])),

                    num_parallel_calls=AUTOTUNE)

test_ds = test_ds.map(lambda path, label: tuple(tf.py_function(cv2_imread, [path, label], [tf.uint16, label.dtype])),

                      num_parallel_calls=AUTOTUNE)



# map the TensorFlow transformation function - no need to wrap

train_ds = train_ds.map(tf_cleanup, num_parallel_calls=AUTOTUNE)

val_ds = val_ds.map(tf_cleanup, num_parallel_calls=AUTOTUNE)

test_ds = test_ds.map(tf_cleanup, num_parallel_calls=AUTOTUNE)
def rotate_augmentation(img, label):

    # rotate 0, 90, 180, or 270 degrees with 25% probability for each

    img = tf.image.rot90(img, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32, seed=1111))

    return img, label



def flip_augmentation(img, label):

    # flip with 50% probability for left-right and up-down

    img = tf.image.random_flip_left_right(img, seed=2222)

    img = tf.image.random_flip_up_down(img, seed=3333)

    return img, label



# map the augmentations, creating a new Dataset

augmented_train_ds = train_ds.map(rotate_augmentation, num_parallel_calls=AUTOTUNE)

augmented_train_ds = augmented_train_ds.map(flip_augmentation, num_parallel_calls=AUTOTUNE)



augmented_val_ds = val_ds.map(rotate_augmentation, num_parallel_calls=AUTOTUNE)

augmented_val_ds = augmented_val_ds.map(flip_augmentation, num_parallel_calls=AUTOTUNE)



# concatenate the augmented and original datasets

train_ds = train_ds.concatenate(augmented_train_ds)

val_ds = val_ds.concatenate(augmented_val_ds)
# double the number of samples in the training and validation splits, due to our augmentation procedure

n_train = len(train_labels)*2

n_val = len(val_labels)*2

n_test = len(test_labels)



# shuffle over the entire dataset, seeding the shuffling for reproducible results

train_ds = train_ds.shuffle(n_train, seed=2019, reshuffle_each_iteration=True)

val_ds = val_ds.shuffle(n_val, seed=2019, reshuffle_each_iteration=True)

test_ds = test_ds.shuffle(n_test, seed=2019, reshuffle_each_iteration=True)
n_train_check = 0

for element in train_ds:

    n_train_check = n_train_check + 1

print(n_train_check)
n_val_check = 0

for element in val_ds:

    n_val_check = n_val_check + 1

print(n_val_check)
n_test_check = 0

for element in test_ds:

    n_test_check = n_test_check + 1

print(n_test_check)
# check that the image was read in correctly

for image, label in train_ds.take(1):

    print("image shape : ", image.numpy().shape)

    print("label       : ", label.numpy())
fig, ax = plt.subplots(nrows=4, ncols=10, sharex=True, sharey=True, figsize=(20,10))



i = 0



for image, label in train_ds.take(10):

    ax[0,i].imshow(image[:,:,0])

    ax[0,i].set_title('{} - {}'.format(label.numpy(), 'R'))

    ax[1,i].imshow(image[:,:,1])

    ax[1,i].set_title('{} - {}'.format(label.numpy(), 'G'))

    ax[2,i].imshow(image[:,:,2])

    ax[2,i].set_title('{} - {}'.format(label.numpy(), 'B'))

    ax[3,i].imshow(image)

    ax[3,i].set_title('{} - {}'.format(label.numpy(), 'RGB'))

    

    i = i+1
fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(20,15))



ax[0,0].set_xlim(0,1)



i = 0



for image, label in train_ds.take(3):

    ax[i,0].hist(image[:,:,0].numpy().flatten())

    ax[i,0].set_title('{} - {}'.format(label.numpy(), 'R'))

    ax[i,1].hist(image[:,:,1].numpy().flatten())

    ax[i,1].set_title('{} - {}'.format(label.numpy(), 'G'))

    ax[i,2].hist(image[:,:,2].numpy().flatten())

    ax[i,2].set_title('{} - {}'.format(label.numpy(), 'B'))

    

    i = i+1
BATCH_SIZE = 32



train_batches_ds = train_ds.batch(BATCH_SIZE)

val_batches_ds = val_ds.batch(BATCH_SIZE)

test_batches_ds = test_ds.batch(BATCH_SIZE)
for image_batch, label_batch in train_batches_ds.take(1):

    print(image_batch.shape)

    print(label_batch.numpy())
IMG_SHAPE = (128, 128, 3)



# create the base model from the pre-trained model VGG16

# note that, if using a Kaggle server, internet has to be turned on

pretrained_model = tf.keras.applications.vgg16.VGG16(input_shape=IMG_SHAPE,

                                                     include_top=False,

                                                     weights='imagenet')



# freeze the convolutional base

pretrained_model.trainable = False
feature_batch = pretrained_model(image_batch)

print(feature_batch.shape)
pretrained_model.summary()
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()



feature_batch_average = global_average_layer(feature_batch)

print(feature_batch_average.shape)
# set the initializers with a seed for reproducible behaviour

prediction_layer = tf.keras.layers.Dense(1,

                                         kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1992),

                                         bias_initializer=tf.keras.initializers.GlorotUniform(seed=1992))



prediction_batch = prediction_layer(feature_batch_average)

print(prediction_batch.shape)
model = tf.keras.Sequential([pretrained_model,

                             global_average_layer,

                             prediction_layer])
base_learning_rate = 0.0001

model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),

              loss='binary_crossentropy',

              metrics=['accuracy'])
model.summary()
initial_epochs = 15

steps_per_epoch = n_train//BATCH_SIZE

validation_steps = 20



loss0, accuracy0 = model.evaluate(val_batches_ds, steps=validation_steps)
history = model.fit(train_batches_ds,

                    epochs=initial_epochs,

                    validation_data=val_batches_ds,

                    validation_steps=validation_steps)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,8), sharex=True)



x_plot = np.arange(1, initial_epochs+1)



ax[0].plot(x_plot, acc, '+-', label='training')

ax[0].plot(x_plot, val_acc, '+-', label='validation')

ax[0].legend()

ax[0].set_ylabel('accuracy')

ax[0].set_ylim(0.5, 1)

ax[0].grid(ls='--', c='C7')

ax[0].set_title('accuracy')



ax[1].plot(x_plot, loss, '+-', label='training')

ax[1].plot(x_plot, val_loss, '+-', label='validation')

ax[1].legend()

ax[1].set_ylabel('cross entropy')

ax[1].set_ylim(0, 1)

ax[1].grid(ls='--', c='C7')

ax[1].set_title('loss')

ax[1].set_xlabel('epoch')



plt.show()
# unfreeze the layers

pretrained_model.trainable = True



# let's take a look to see how many layers are in the base model

print("Number of layers in the pre-trained model: ", len(pretrained_model.layers))
# fine-tune from this layer onwards

fine_tune_at = 15



# freeze all the layers before the `fine_tune_at` layer

for layer in pretrained_model.layers[:fine_tune_at]:

  layer.trainable =  False
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate/10),

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.summary()
len(model.trainable_variables)
fine_tune_epochs = 25

total_epochs =  initial_epochs + fine_tune_epochs



history_fine = model.fit(train_batches_ds,

                         epochs=total_epochs,

                         initial_epoch=history.epoch[-1]+1,

                         validation_data=val_batches_ds,

                         validation_steps=validation_steps)
acc += history_fine.history['accuracy']

val_acc += history_fine.history['val_accuracy']



loss += history_fine.history['loss']

val_loss += history_fine.history['val_loss']



fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,8), sharex=True)



x_plot = np.arange(1, total_epochs+1)



ax[0].plot(x_plot, acc, '+-', label='training')

ax[0].plot(x_plot, val_acc, '+-', label='validation')

ax[0].legend()

ax[0].set_ylabel('accuracy')

ax[0].set_ylim(0.5, 1)

ax[0].grid(ls='--', c='C7')

ax[0].set_title('accuracy')

ax[0].axvline(initial_epochs, c='C7', ls='--')



ax[1].plot(x_plot, loss, '+-', label='training')

ax[1].plot(x_plot, val_loss, '+-', label='validation')

ax[1].legend()

ax[1].set_ylabel('cross entropy')

ax[1].set_ylim(0, 1)

ax[1].grid(ls='--', c='C7')

ax[1].set_title('loss')

ax[1].set_xlabel('epoch')

ax[1].axvline(initial_epochs, c='C7', ls='--')



plt.show()
val_loss, val_accuracy = model.evaluate(val_batches_ds)
test_loss, test_accuracy = model.evaluate(test_batches_ds)