# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.model_selection import train_test_split



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# # Google API for MNIST data

# # Sample Code of TensorFlow API

# DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'



# path = tf.keras.utils.get_file('mnist.npz', DATA_URL)

# with np.load(path) as data:

#   tf_train_examples = data['x_train']

#   #plt.imshow(tf_train_examples[1])

#   print('train_data_shape: ', tf_train_examples.shape)

#   tf_train_labels = data['y_train']

#   print('train_label_shape: ', tf_train_labels.shape)

#   tf_test_examples = data['x_test']

#   print('test_data_shape: ', tf_test_examples.shape)

#   tf_test_labels = data['y_test']

#   print('test_label_shape: ', tf_test_labels.shape)

    

# # Display same files

# fig = plt.figure()

# for i in range(1, 10):

#     ax = fig.add_subplot(3, 3, i)

#     ax.imshow(tf_train_examples[i])

#     ax.set_title(tf_train_labels[i])

# fig.show()

    

# tf_train_dataset = tf.data.Dataset.from_tensor_slices((tf_train_examples, tf_train_labels))

# tf_test_dataset = tf.data.Dataset.from_tensor_slices((tf_test_examples, tf_test_labels))



# BATCH_SIZE = 64

# SHUFFLE_BUFFER_SIZE = 100



# tf_train_dataset = tf_train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

# tf_test_dataset = tf_test_dataset.batch(BATCH_SIZE)



# # Construct a model for training

# tf_model = tf.keras.Sequential([

#     tf.keras.layers.Flatten(input_shape=(28,28)),

#     tf.keras.layers.Dense(128, activation='relu'),

#     tf.keras.layers.Dense(10)

# ])



# tf_model.compile(optimizer=tf.keras.optimizers.RMSprop(),

#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

#               metrics=['sparse_categorical_accuracy'])

# tf_model.fit(tf_train_dataset, epochs=10)

# tf_model.evaluate(tf_test_dataset)
# Load Train and Competition Dataset

train_raw_dataset = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

X_train_raw = train_raw_dataset.drop('label', axis = 1) / 255.0

# major improvement with categorical output

y_train_raw = tf.keras.utils.to_categorical(train_raw_dataset['label'], num_classes=10, dtype='uint8')



competition_raw_dataset = pd.read_csv('/kaggle/input/digit-recognizer/test.csv') / 255.0
# Resize image  

IMG_SIZE = 28

X_train_raw = X_train_raw.values.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

competition_raw_dataset = competition_raw_dataset.to_numpy().reshape(-1, IMG_SIZE, IMG_SIZE, 1)



print("Training set (images) shape: {shape}".format(shape=X_train_raw.shape))

print("Training set (labels) shape: {shape}".format(shape=y_train_raw.shape))

print("Competition set (images) shape: {shape}".format(shape=competition_raw_dataset.shape))
# Plot Images

PLOT_SIZE = 5

plt.figure(figsize=(10,10))

for i in range(PLOT_SIZE * PLOT_SIZE):

    plt.subplot(PLOT_SIZE, PLOT_SIZE, i + 1)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(X_train_raw[i][:, :, 0], cmap=plt.cm.binary)

    plt.xlabel(y_train_raw[i].tolist().index(1))

plt.show()
# Define CONSTANTS

RANDOM_STATE = 7

TEST_SIZE = 0.1



X_train, X_test, y_train, y_test = train_test_split(X_train_raw, y_train_raw, 

                                                    test_size=TEST_SIZE, random_state = RANDOM_STATE)



train_dataset = tf.data.Dataset.from_tensor_slices((tf.dtypes.cast(X_train, tf.float64), tf.dtypes.cast(y_train, tf.uint8)))

test_dataset = tf.data.Dataset.from_tensor_slices((tf.dtypes.cast(X_test, tf.float64), tf.dtypes.cast(y_test, tf.uint8)))

competition_dataset = tf.data.Dataset.from_tensor_slices(tf.dtypes.cast(competition_raw_dataset, tf.float64))



print(train_dataset)

print(test_dataset)

print(competition_dataset)
BATCH_SIZE = 64

SHUFFLE_BUFFER_SIZE = 100



train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

test_dataset = test_dataset.batch(BATCH_SIZE)

competition_dataset = competition_dataset.batch(BATCH_SIZE)
# Construct a model for training

model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding = 'same', input_shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding = 'same'),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding = 'same'),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding = 'same'),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(10, activation='softmax')

])



model.summary()
# Define the optimizer

# https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, 

                            momentum=0.0, epsilon=1e-08, centered=False)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])



# Data augmentation to prevent overfitting

datagen = tf.keras.preprocessing.image.ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)



# Progressively reduce learning rate

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, 

                                                               factor=0.5, min_lr=0.00001)

earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,

                             patience=1, verbose=1, mode='auto')
history = model.fit(datagen.flow(X_train,y_train, batch_size=BATCH_SIZE), 

                    epochs=20, callbacks=[learning_rate_reduction, earlystopper])
model.evaluate(test_dataset)
# Predict for competition

predictions = model.predict_classes(competition_dataset)

indices = np.arange(1, competition_raw_dataset.shape[0]+1)



# Save output

from datetime import datetime

timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')



output = pd.DataFrame({'ImageId': indices, 'Label': predictions})

output.to_csv('submission_random_forest_' + timestamp + '.csv', index=False)

print("Your submission was successfully saved!")
## To DO

# 1. To show hidden layers 

a = model.get_layer(index=1).get_weights()

print(a[0].shape)



# Plot Images

PLOT_SIZE = 5

plt.figure(figsize=(10,10))

for i in range(PLOT_SIZE * PLOT_SIZE):

    plt.subplot(PLOT_SIZE, PLOT_SIZE, i + 1)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(a[0][:,:,i,0])

    plt.xlabel(a[1][i])

plt.show()



# 2. To visualize learning process