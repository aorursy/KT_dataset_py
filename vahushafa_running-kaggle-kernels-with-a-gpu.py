# Imports for Deep Learning

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, BatchNormalization, UpSampling2D

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator

from keras import regularizers

from keras.losses import categorical_crossentropy, binary_crossentropy, mean_squared_error



# ensure consistency across runs

from numpy.random import seed

seed(1)

from tensorflow import set_random_seed

set_random_seed(2)



# Imports to view data

import cv2

from glob import glob

from matplotlib import pyplot as plt

from numpy import floor

import random



def plot_three_samples(letter):

    print("Samples images for letter " + letter)

    base_path = '../input/asl_alphabet_train/asl_alphabet_train/'

    img_path = base_path + letter + '/**'

    path_contents = glob(img_path)

    

    plt.figure(figsize=(16,16))

    imgs = random.sample(path_contents, 3)

    plt.subplot(131)

    plt.imshow(cv2.imread(imgs[0]))

    plt.subplot(132)

    plt.imshow(cv2.imread(imgs[1]))

    plt.subplot(133)

    plt.imshow(cv2.imread(imgs[2]))

    return



plot_three_samples('A')
plot_three_samples('B')
data_dir = "../input/asl_alphabet_train/asl_alphabet_train"

target_size = (224, 224)

target_dims = (224, 224, 3) # add channel for RGB

n_classes = 29

val_frac = 0.1

batch_size = 64



data_augmentor = ImageDataGenerator(samplewise_center=True, 

                                    samplewise_std_normalization=True, 

                                    validation_split=val_frac)



train_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, shuffle=True, subset="training")

val_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, subset="validation")

# Model Specification
test_dir = "../input/asl_alphabet_test/asl_alphabet_test"

labels = train_generator.class_indices

import numpy as np

from os import listdir



test_imgs = []

y = []

for path in listdir(test_dir):

    img = cv2.imread(test_dir + '/'+ path)

    img = cv2.resize(img, target_size)

    test_imgs.append(img)

    y.append( [1 if j == labels[path.split('_')[0]] else 0 for j in range(n_classes)] )

    

test_generator = data_augmentor.flow(x=np.array(test_imgs), y=np.array(y), batch_size=batch_size, shuffle=False)

import matplotlib.pyplot as plt

def plot_history(history):

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title('Model accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.show()

    

# plot_history(history)
from keras.applications.inception_v3 import InceptionV3

from keras.models import Model

base_model = InceptionV3(include_top = False, weights='imagenet', input_shape = target_dims, pooling='max')

# f = Flatten()(base_model.output)

dr = Dropout(0.5)(base_model.output)

d1 = Dense(512, activation = 'relu', kernel_regularizer = regularizers.l1(0.001))(dr)

d2 = Dense(n_classes, activation = 'softmax')(d1)

vggmodel = Model(base_model.input, d2)

vggmodel.compile(optimizer = 'adam', loss = categorical_crossentropy, metrics = ["accuracy"])

vggmodel.summary()
vgghistory=vggmodel.fit_generator(train_generator, epochs=5, validation_data=val_generator)
vggmodel.evaluate_generator(test_generator, steps = 1, verbose = 1)
vggmodel.save("vggmodel.h5")
plot_history(vgghistory)