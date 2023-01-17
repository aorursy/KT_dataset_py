import numpy as np # linear algebra



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
with open('../input/10-monkey-species/monkey_labels.txt') as f:

    print("".join(f.readlines()))
print(os.listdir("../input/10-monkey-species/training/training"))
print(os.listdir("../input/10-monkey-species/training/training/n0"))
print(os.listdir("../input/10-monkey-species/validation/validation"))
# compute readable labels

real_labels = {

    "n0": "mantled_howler",

    "n1": "patas_monkey",

    "n2": "bald_uakari",

    "n3": "japanese_macaque",

    "n4": "pygmy_marmoset",

    "n5": "white_headed_capuchin",

    "n6": "silvery_marmoset",

    "n7": "common_squirrel_monkey",

    "n8": "black_headed_night_monkey",

    "n9": "nilgiri_langur"}
import random

import shutil
# separate validation data into test - validation

num_val_files = 16  # number of validation files per label



test_dir = './10-monkey-species-test'

val_dir = './10-monkey-species-val'

orig_val_dir = '../input/10-monkey-species/validation/validation'

os.mkdir(test_dir)

os.mkdir(val_dir)



for i in range(10):

    folder_name = 'n{}'.format(i)  # subfolder name for the label

    

    src_folder = os.path.join(orig_val_dir, folder_name)

    dst_val_dir = os.path.join(val_dir, folder_name)

    dst_test_dir = os.path.join(test_dir, folder_name)

    

    # pick files:

    # make a list and sort it randomly

    files = os.listdir(src_folder)

    random.shuffle(files)



    os.mkdir(dst_val_dir)

    for val_file in files[:num_val_files]:

        src_file = os.path.join(src_folder, val_file)

        dst_file = os.path.join(dst_val_dir, val_file)

        shutil.copyfile(src_file, dst_file)

    print(dst_val_dir, os.listdir(dst_val_dir))

    

    os.mkdir(dst_test_dir)

    for test_file in files[num_val_files:]:

        src_file = os.path.join(src_folder, test_file)

        dst_file = os.path.join(dst_test_dir, test_file)

        shutil.copyfile(src_file, dst_file)

    print(dst_test_dir, os.listdir(dst_test_dir))
import tensorflow as tf

from tensorflow import keras



print("TensorFlow", tf.__version__)

print("Keras", keras.__version__)
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from IPython import display
# display pictures

for i in range(4):

    folder = '../input/10-monkey-species/training/training/n{}'.format(i)

    first_image = os.listdir(folder)[0]

    print("n{} {}".format(i, real_labels["n{}".format(i)]))

    display.display(display.Image(os.path.join(folder, first_image)))
# construct the model

num_classes = 10

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



# use Resnet without top for model

model = Sequential()

model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))

# top layer

model.add(Dense(num_classes, activation='softmax'))



# Indicate whether the first layer should be trained/changed or not.

model.layers[0].trainable = False
# describe the model

model.summary()
model.compile(

    optimizer='sgd',

    loss='categorical_crossentropy',

    metrics=['accuracy'])
# data augmentation

image_size = 224  # necessary for ResNet50

num_samples = 1098

batch_size = 20



data_generator = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

    zoom_range = 0.1, # Randomly zoom image

    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

    horizontal_flip=True,  # randomly flip images

    vertical_flip=False)  # randomly flip images)



# for validation and test only

data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)



train_generator = data_generator.flow_from_directory(

                                        directory='../input/10-monkey-species/training/training',

                                        target_size=(image_size, image_size),

                                        batch_size=batch_size,

                                        class_mode='categorical')



validation_generator = data_generator_no_aug.flow_from_directory(

                                        directory=val_dir,

                                        target_size=(image_size, image_size),

                                        class_mode='categorical')
# train the model

fit_stats = model.fit_generator(train_generator,

                                steps_per_epoch=num_samples // batch_size,

                                epochs=5,

                                validation_data=validation_generator,

                                validation_steps=1)
import matplotlib.pyplot as plt
# display statistics

# code inspired from Olga Belitskaya's notebook

# https://www.kaggle.com/olgabelitskaya/handwritten-letters

plt.figure(figsize=(18, 12))

    

plt.subplot(211)

plt.plot(fit_stats.history['loss'], color='slategray', label = 'train')

plt.plot(fit_stats.history['val_loss'], color='#4876ff', label = 'valid')

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend()

plt.title('Loss Function');  



plt.subplot(212)

plt.plot(fit_stats.history['acc'], color='slategray', label = 'train')

plt.plot(fit_stats.history['val_acc'], color='#4876ff', label = 'valid')

plt.xlabel("Epochs")

plt.ylabel("Accuracy")    

plt.ylim(0.0, 1.0)

plt.legend()

plt.title('Accuracy');
# evaluate on test data

test_generator = data_generator_no_aug.flow_from_directory(

    directory=test_dir,

    target_size=(image_size, image_size),

    class_mode='categorical')

score = model.evaluate_generator(test_generator)

print(score)

print("Accuracy: {:.2f}%".format(100 * score[1]))
# delete output data

shutil.rmtree(val_dir)

shutil.rmtree(test_dir)