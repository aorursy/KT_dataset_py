import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns 

sns.set()

from scipy import misc

import imageio as im

import os

import warnings

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import itertools

warnings.filterwarnings('ignore')

%config InlineBackend.figure_format = 'retina'
img = im.imread('../input/basicshapes/shapes/circles/drawing(10).png')

plt.imshow(img, cmap='gray')
def make_labels(directory, data=[], y_hat=[], label=0):

    for root, dirs, files in os.walk(directory):

        for file in files:

            img = im.imread(directory+file)

            data.append(img)

        y_hat = [label] * len(data)

    return np.array(data), np.array(y_hat)
circles, y_circles = [], []

circles, y_circles = make_labels('../input/basicshapes/shapes/circles/', data=circles, y_hat=y_circles)



squares, y_squares = [], []

squares, y_squares = make_labels('../input/basicshapes/shapes/squares/', data=squares, y_hat=y_squares, label=1)



triangles, y_triangles = [], []

triangles, y_triangles = make_labels('../input/basicshapes/shapes/triangles/', data=triangles, y_hat=y_triangles, label=2)
print(circles.shape, squares.shape, triangles.shape)

print(y_circles.shape, y_squares.shape, y_triangles.shape)
%matplotlib inline

import glob

import matplotlib

from matplotlib import pyplot as plt

import matplotlib.image as mpimg

import numpy as np

import imageio as im

from keras import models

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint
images = []

for img_path in glob.glob('../input/basicshapes/shapes/circles/*.png'):

    images.append(mpimg.imread(img_path))

plt.figure(figsize=(20,10))

columns = 5

for i, image in enumerate(images):

    plt.subplot(len(images) / columns + 1, columns, i + 1)

    plt.imshow(image)
images = []

for img_path in glob.glob('../input/basicshapes/shapes/squares/*.png'):

    images.append(mpimg.imread(img_path))

plt.figure(figsize=(20,10))

columns = 5

for i, image in enumerate(images):

    plt.subplot(len(images) / columns + 1, columns, i + 1)

    plt.imshow(image)

plt.show()
images = []

for img_path in glob.glob('../input/basicshapes/shapes//triangles/*.png'):

    images.append(mpimg.imread(img_path))



plt.figure(figsize=(20,10))

columns = 5

for i, image in enumerate(images):

    plt.subplot(len(images) / columns + 1, columns, i + 1)

    plt.imshow(image)
img = im.imread('../input/basicshapes/shapes/circles/drawing(40).png')

img.shape
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), padding='same', input_shape = (28, 28, 3), activation = 'relu'))

classifier.add(Conv2D(32, (3, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Dropout(0.5)) # antes era 0.25



# Adding a second convolutional layer

classifier.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))

classifier.add(Conv2D(64, (3, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Dropout(0.5)) # antes era 0.25



# Adding a third convolutional layer

classifier.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))

classifier.add(Conv2D(64, (3, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Dropout(0.5)) # antes era 0.25



# Step 3 - Flattening

classifier.add(Flatten())



# Step 4 - Full connection

classifier.add(Dense(units = 512, activation = 'relu'))

classifier.add(Dropout(0.5)) 

classifier.add(Dense(units = 4, activation = 'softmax'))
classifier.summary()
full_multiclass_report(model_gen,

                       X_test,

                       y_test,

                       ['circles', 'squares', 'triangles'])
classifier.summary()
classifier.compile(optimizer = 'rmsprop',

                   loss = 'categorical_crossentropy', 

                   metrics = ['accuracy'])
train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('../input/basicshapes/shapes/',

                                                 target_size = (28, 28),

                                                 batch_size = 16,

                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('../input/basicshapes/shapes/',

                                            target_size = (28, 28),

                                            batch_size = 16,

                                            class_mode = 'categorical')
checkpointer = ModelCheckpoint(filepath="/kaggle/working/best_weights.hdf5", 

                               monitor = 'val_acc',

                               verbose=1, 

                               save_best_only=True)
history = classifier.fit_generator(training_set,

                                   steps_per_epoch = 100,

                                   epochs = 1,

                                   callbacks=[checkpointer],

                                   validation_data = test_set,

                                   validation_steps = 50)
classifier.save('/kaggle/working/shapes_cnn.h5')
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
img_path = '../input/basicshapes/shapes/squares/drawing(15).png'

from keras.preprocessing import image

imgs = image.load_img(img_path, target_size=(28, 28))

img_tensor = image.img_to_array(imgs)

img_tensor = np.expand_dims(img_tensor, axis=0)

img_tensor /= 255.



plt.imshow(img_tensor[0])

plt.show()



print(img_tensor.shape)
x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

images = np.vstack([x])

classes = classifier.predict_classes(images, batch_size=10)

print("Predicted class is:",classes)
# Extracts the outputs of the top 12 layers

layer_outputs = [layer.output for layer in classifier.layers[:12]] 

# Creates a model that will return these outputs, given the model input

activation_model = models.Model(inputs=classifier.input, outputs=layer_outputs) 
# Returns a list of five Numpy arrays: one array per layer activation

activations = activation_model.predict(img_tensor) 
first_layer_activation = activations[0]

print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
layer_names = []

for layer in classifier.layers[:12]:

    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot

    

images_per_row = 16



for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps

    n_features = layer_activation.shape[-1] # Number of features in the feature map

    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).

    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix

    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols): # Tiles each filter into a big horizontal grid

        for row in range(images_per_row):

            channel_image = layer_activation[0,

                                             :, :,

                                             col * images_per_row + row]

            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable

            channel_image /= channel_image.std()

            channel_image *= 64

            channel_image += 128

            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            display_grid[col * size : (col + 1) * size, # Displays the grid

                         row * size : (row + 1) * size] = channel_image

    scale = 1. / size

    plt.figure(figsize=(scale * display_grid.shape[1],

                        scale * display_grid.shape[0]))

    plt.title(layer_name)

    plt.grid(False)

    plt.imshow(display_grid, aspect='auto', cmap='viridis')
from keras.applications.vgg16 import VGG16

classifier = VGG16(weights='imagenet', include_top=True)
from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.applications.vgg16 import preprocess_input

from keras.applications.vgg16 import decode_predictions

from keras.preprocessing import image

import numpy as np

# load an image from file

image = load_img('/kaggle/input/dogimage/cat.jpg', target_size=(224, 224))

plt.imshow(image)

# convert the image pixels to a numpy array

image = img_to_array(image)

# reshape data for the model

image = np.expand_dims(image, axis=0)

# prepare the image for the VGG model



img_tensor = preprocess_input(image)



block4_pool_features = classifier.predict(img_tensor)


# convert the probabilities to class labels

label = decode_predictions(block4_pool_features)

# retrieve the most likely result, e.g. highest probability

label
from keras.applications.resnet50 import ResNet50

classifier = ResNet50(weights='imagenet', include_top=True)
classifier.summary()
from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.applications.resnet50 import preprocess_input

from keras.applications.resnet50 import decode_predictions

from keras.preprocessing import image

import numpy as np

# load an image from file

image = load_img('/kaggle/input/dogimage/cat.jpg', target_size=(224, 224))

# convert the image pixels to a numpy array

image = img_to_array(image)

# reshape data for the model

image = np.expand_dims(image, axis=0)

# prepare the image for the VGG model

img_tensor = preprocess_input(image)



block4_pool_features = model.predict(image)
# predict the probability across all output classes

yhat = classifier.predict(image)

# convert the probabilities to class labels

label = decode_predictions(yhat)

# retrieve the most likely result, e.g. highest probability

label
# Extracts the outputs of the top 12 layers

layer_outputs = [layer.output for layer in classifier.layers[:18]] 

# Creates a model that will return these outputs, given the model input

activation_model = models.Model(inputs=classifier.input, outputs=layer_outputs) 
# Returns a list of five Numpy arrays: one array per layer activation

activations = activation_model.predict(img_tensor) 
first_layer_activation = activations[0]

print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 0], cmap='viridis')
layer_names = []

for layer in classifier.layers[:17]:

    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot

    

images_per_row = 8



for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps

    n_features = layer_activation.shape[-1] # Number of features in the feature map

    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).

    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix

    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols): # Tiles each filter into a big horizontal grid

        for row in range(images_per_row):

            channel_image = layer_activation[0,

                                             :, :,

                                             col * images_per_row + row]

            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable

            channel_image /= channel_image.std()

            channel_image *= 64

            channel_image += 128

            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            display_grid[col * size : (col + 1) * size, # Displays the grid

                         row * size : (row + 1) * size] = channel_image

    scale = 1. / size

    plt.figure(figsize=(scale * display_grid.shape[1],

                        scale * display_grid.shape[0]))

    plt.title(layer_name)

    plt.grid(False)

    plt.imshow(display_grid, aspect='auto', cmap='viridis')