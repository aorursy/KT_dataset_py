import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook
!ls -lha ../input/planet-understanding-the-amazon-from-space
input_path = '../input/planet-understanding-the-amazon-from-space/'



print('File sizes:')

print('')

for f in os.listdir(input_path):

    if not os.path.isdir(input_path + f):

        print(f.ljust(30) + str(round(os.path.getsize(input_path + f) / 1000000, 2)) + 'MB')

    else:

        sizes = [os.path.getsize(input_path+f+'/'+x)/1000000 for x in os.listdir(input_path + f)]

        print(f.ljust(30) + str(round(sum(sizes), 2)) + 'MB' + ' ({} files)'.format(len(sizes)))
train_df = pd.read_csv(input_path + 'train_v2.csv')

train_df.head()
train_df.info()
# build list with unique labels

label_list = []

for tag_str in train_df['tags'].values:

    labels = tag_str.split(' ')

    for label in labels:

        if label not in label_list:

            label_list.append(label)

            

label_list
# binarize the labels

for label in label_list:

    train_df[label] = train_df['tags'].apply(lambda x : 1 if label in x.split(' ') else 0)

    

train_df.head()
train_df[label_list].sum().sort_values().plot.bar()

plt.show()
# a function to quickly plot the co-occurences

def show_cooccurence_matrix(labels):

    numeric_df = train_df[labels]

    co_matrix = numeric_df.T.dot(numeric_df)

    

    fig, ax = plt.subplots(figsize=(7,7))

    im = ax.imshow(co_matrix)

    ax.set_xticks(np.arange(len(labels)))

    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, rotation=45, ha='right', rotation_mode='anchor')

    ax.set_yticklabels(labels)

    cbar = fig.colorbar(im, ax=ax) # not sure why this isn't working...

    plt.show(fig)

    

# compute the co-ocurrence matrix

show_cooccurence_matrix(label_list)
weather_labels = ['clear', 'partly_cloudy', 'haze', 'cloudy']

show_cooccurence_matrix(weather_labels)
land_labels = ['primary', 'agriculture', 'water', 'cultivation', 'habitation']

show_cooccurence_matrix(land_labels)
import cv2



# read it in unchanged, to make sure we aren't losing any information

img = cv2.imread(input_path + 'train-jpg/{}.jpg'.format(train_df['image_name'][0]), cv2.IMREAD_UNCHANGED)

np.shape(img)
jpg_channels = ['B','G','R']



fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(20,12))



ax = ax.flatten()



for i in range(6):

    img = cv2.imread(input_path + 'train-jpg/{}.jpg'.format(train_df['image_name'][i]), cv2.IMREAD_UNCHANGED)

    ax[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    ax[i].set_title('{} - {}'.format(train_df['image_name'][i], train_df['tags'][i]))

    

plt.show()
type(img[0,0,0])
jpg_channel_colors = ['b','g','r']



fig, ax = plt.subplots()



for i in range(len(jpg_channels)):

    ax.hist(img[:,:,i].flatten(), bins=100, range=[np.min(img), np.max(img)],

            label=jpg_channels[i], color=jpg_channel_colors[i], alpha=0.5)

    ax.legend()

    

ax.set_xlim(0,255)

    

plt.show(fig)
# read it in unchanged, to make sure we aren't losing any information

img = cv2.imread(input_path + 'train-tif-v2/{}.tif'.format(train_df['image_name'][0]), cv2.IMREAD_UNCHANGED)

np.shape(img)
tif_channels = ['B','G','R','NIR']



fig, ax = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(20,15))



ax = ax.flatten()



img = cv2.imread(input_path + 'train-tif-v2/{}.tif'.format(train_df['image_name'][1]), cv2.IMREAD_UNCHANGED)



for i in range(len(tif_channels)):

    ax[i].imshow(img[:,:,i])

    ax[i].set_title('{} - {}'.format(train_df['image_name'][1], tif_channels[i]))

    

plt.show()
type(img[0,0,0])
tif_channel_colors = ['b','g','r','magenta']



fig, ax = plt.subplots()



for i in range(len(tif_channels)):

    ax.hist(img[:,:,i].flatten(), bins=100, range=[np.min(img), np.max(img)],

            label=tif_channels[i], color=tif_channel_colors[i], alpha=0.5)

    ax.legend()

    

plt.show(fig)
import tensorflow as tf

tf.__version__
# extract image names from the .csv with the labels

path_to_images = train_df['image_name'].copy().values



# convert to path

for i in range(len(path_to_images)):

    path_to_images[i] = input_path + 'train-jpg/' + path_to_images[i] + '.jpg'



path_to_images[:5]
weather_labels_array = train_df[weather_labels].copy().values.astype(bool)



weather_labels_array[:5]
train_df[weather_labels].head()
weather_ds = tf.data.Dataset.from_tensor_slices((path_to_images, weather_labels_array))



# note that the `numpy()` function is required to grab the actual values from the Dataset

for path, label in weather_ds.take(5):

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

    img.set_shape((256, 256, 3))

    # convert to float32, scaling from uint8 (0-255) to float32 (0-1)

    img = tf.image.convert_image_dtype(img, tf.float32)

    # resize the image

    img = tf.image.resize(img, [256, 256])

    return img, label



AUTOTUNE = tf.data.experimental.AUTOTUNE



# map the cv2 wrapper function using `tf.py_function`

weather_ds = weather_ds.map(lambda path, label: tuple(tf.py_function(cv2_imread, [path, label], [tf.uint16, label.dtype])),

                            num_parallel_calls=AUTOTUNE)



# map the TensorFlow transformation function - no need to wrap

weather_ds = weather_ds.map(tf_cleanup, num_parallel_calls=AUTOTUNE)
for image, label in weather_ds.take(1):

    print("image shape : ", image.numpy().shape)

    print("label       : ", label.numpy())
fig, ax = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(20,15))



i = 0



for image, label in weather_ds.take(3):

    ax[i,0].imshow(image[:,:,0])

    ax[i,0].set_title('{} - {}'.format(label.numpy(), 'R'))

    ax[i,1].imshow(image[:,:,1])

    ax[i,1].set_title('{} - {}'.format(label.numpy(), 'G'))

    ax[i,2].imshow(image[:,:,2])

    ax[i,2].set_title('{} - {}'.format(label.numpy(), 'B'))

    ax[i,3].imshow(image)

    ax[i,3].set_title('{} - {}'.format(label.numpy(), 'RGB'))

    

    i = i+1
fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(20,15))



ax[0,0].set_xlim(0,1)



i = 0



for image, label in weather_ds.take(3):

    ax[i,0].hist(image[:,:,0].numpy().flatten())

    ax[i,0].set_title('{} - {}'.format(label.numpy(), 'R'))

    ax[i,1].hist(image[:,:,1].numpy().flatten())

    ax[i,1].set_title('{} - {}'.format(label.numpy(), 'G'))

    ax[i,2].hist(image[:,:,2].numpy().flatten())

    ax[i,2].set_title('{} - {}'.format(label.numpy(), 'B'))

    

    i = i+1
n_all = len(path_to_images)

n_train = int(n_all * 0.8)

n_val = n_all - n_train



# shuffle the Dataset

SHUFFLE_BUFFER_SIZE = 1000

weather_ds = weather_ds.shuffle(SHUFFLE_BUFFER_SIZE)



# n_train will be used for training, the rest will be used for validation

train_weather_ds = weather_ds.take(n_train)

val_weather_ds = weather_ds.skip(n_train)
BATCH_SIZE = 32



train_weather_batches_ds = train_weather_ds.batch(BATCH_SIZE)

val_weather_batches_ds = val_weather_ds.batch(BATCH_SIZE)
for image_batch, label_batch in train_weather_batches_ds.take(1):

    print(image_batch.shape)
IMG_SHAPE = (256, 256, 3)



# create the base model from the pre-trained model VGG16

# note that, if using a Kaggle server, internet has to be turned on

weather_VGG16 = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,

                                            include_top=False,

                                            weights='imagenet')



# freeze the convolutional base

weather_VGG16.trainable = False



# look at the model architecture

weather_VGG16.summary()