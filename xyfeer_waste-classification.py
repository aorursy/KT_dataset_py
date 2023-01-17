# importing libraries
import numpy as np
from PIL import Image
import cv2 
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Dimesion of images to be use. Can choose anything as a different dimenion will be resized to this.
ht = 64
wth = 64

# Important hyperparameters
batch_size = 32
no_of_epochs = 4
# num_classes = 2
# The directory for training dataset and validation dataset to be specified in data pipeline
train_data_dir = "/kaggle/input/waste-classification-data/dataset/DATASET/TRAIN"
test_data_dir="/kaggle/input/waste-classification-data/dataset/DATASET/TEST"

# Rescaling of image datasets 
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
# This is the augmentation configuration we will use for training
train_ds = train_datagen.flow_from_directory(
  train_data_dir,
  target_size=(ht, wth),
  batch_size=batch_size,
  class_mode = 'binary'
)
# This is the augmentation configuration we will use for testing
test_ds = test_datagen.flow_from_directory(
  test_data_dir,
  target_size=(ht, wth),
  batch_size=batch_size,
  class_mode = 'binary'
)
from tensorflow.keras import layers

model = tf.keras.Sequential([
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  # layers.Dense(num_classes) // changed this to sigmoid becoz it the correct activation at end layer for classification model.
  layers.Dense(1, activation = 'sigmoid') 
])

model.compile(
  # loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), // I had changed to binary_crossentropy becoz the dataset has 2 class.
  loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit_generator(
  train_ds,
  steps_per_epoch = 706,
  validation_data=test_ds,
  epochs=no_of_epochs,
 validation_steps = 2000
)

from keras.preprocessing import image
# add the image path 
test_image = image.load_img('/kaggle/input/testing-image/recyclable.jpg',
                            target_size = (64, 64))
# add the image path to visualize the image
Image.open('/kaggle/input/testing-image/recyclable.jpg')
# converting image file into array 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
# image prediction
result = model.predict(test_image)
# check the dataset class index
train_ds.class_indices
print(result)
if result[0][0] == 1:
    prediction = 'Recyclable Waste'
else:
    prediction = 'Organic Waste'
# printing the predicted class
print(prediction)