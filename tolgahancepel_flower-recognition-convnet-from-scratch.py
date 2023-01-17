import os

import shutil

from os.path import isfile, join, abspath, exists, isdir, expanduser

from os import listdir, makedirs, getcwd, remove

from pathlib import Path

import numpy as np

import pandas as pd

import seaborn as sns

sns.set_style('darkgrid')

import matplotlib.pyplot as plt

import matplotlib.image as mimg



from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array

from keras import layers

from keras import models

from keras import optimizers
# Check for the directory and if it doesn't exist, make one.

cache_dir = expanduser(join('~', '.keras'))

if not exists(cache_dir):

    makedirs(cache_dir)

    

# make the models sub-directory

models_dir = join(cache_dir, 'models')

if not exists(models_dir):

    makedirs(models_dir)
# original dataset folder, you can see above

input_path = Path('/kaggle/input/flowers-recognition/flowers')

flowers_path = input_path / 'flowers'
# Each species of flower is contained in a separate folder. Get all the sub directories

flower_types = os.listdir(flowers_path)

print("Types of flowers found: ", len(flower_types))

print("Categories of flowers: ", flower_types)
# A list that is going to contain tuples: (species of the flower, corresponding image path)

flowers = []



for species in flower_types:

    # Get all the file names

    all_flowers = os.listdir(flowers_path / species)

    # Add them to the list

    for flower in all_flowers:

        flowers.append((species, str(flowers_path /species) + '/' + flower))



# Build a dataframe        

flowers = pd.DataFrame(data=flowers, columns=['category', 'image'], index=None)

flowers.head()
# feel free to edit "0" (corresponds 0. image)

# flowers['image'][0]
# Let's check how many samples for each category are present

print("Total number of flowers in the dataset: ", len(flowers))

fl_count = flowers['category'].value_counts()

print("Flowers in each category: ")

print(fl_count)
# Let's do some visualization and see how many samples we have for each category



f, axe = plt.subplots(1,1,figsize=(14,6))

sns.barplot(x = fl_count.index, y = fl_count.values, ax = axe)

axe.set_title("Flowers count for each category", fontsize=16)

axe.set_xlabel('Category', fontsize=14)

axe.set_ylabel('Count', fontsize=14)

plt.show()
# Let's visualize some flowers from each category



# A list for storing names of some random samples from each category

random_samples = []



# Get samples fom each category 

for category in fl_count.index:

    samples = flowers['image'][flowers['category'] == category].sample(4).values

    for sample in samples:

        random_samples.append(sample)



# Plot the samples

f, ax = plt.subplots(5,4, figsize=(15,10))

for i,sample in enumerate(random_samples):

    ax[i//4, i%4].imshow(mimg.imread(random_samples[i]))

    ax[i//4, i%4].axis('off')

plt.show()    
# Make a parent directory `data` and two sub directories `train` and `valid`

%mkdir -p data/train

%mkdir -p data/valid



# Inside the train and validation sub=directories, make sub-directories for each catgeory

%cd data

%mkdir -p train/daisy

%mkdir -p train/tulip

%mkdir -p train/sunflower

%mkdir -p train/rose

%mkdir -p train/dandelion



%mkdir -p valid/daisy

%mkdir -p valid/tulip

%mkdir -p valid/sunflower

%mkdir -p valid/rose

%mkdir -p valid/dandelion



%cd ..



# You can verify that everything went correctly using ls command
for category in fl_count.index:

    samples = flowers['image'][flowers['category'] == category].values

    perm = np.random.permutation(samples)

    # Copy first 100 samples to the validation directory and rest to the train directory

    for i in range(100):

        name = perm[i].split('/')[-1]

        shutil.copyfile(perm[i],'./data/valid/' + str(category) + '/'+ name)

    for i in range(101,len(perm)):

        name = perm[i].split('/')[-1]

        shutil.copyfile(perm[i],'./data/train/' + str(category) + '/' + name)
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',

input_shape=(240, 240, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.4))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(5, activation='softmax'))



model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
# Define the generators

batch_size = 32

# this is the augmentation configuration we will use for training

train_datagen = ImageDataGenerator(

        rescale=1./255,

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



# this is the augmentation configuration we will use for testing:

# only rescaling

test_datagen = ImageDataGenerator(rescale=1./255)



# this is a generator that will read pictures found in

# subfolers of 'data/train', and indefinitely generate

# batches of augmented image data

train_generator = train_datagen.flow_from_directory(

        'data/train',  # this is the target directory

        target_size=(240, 240),  # all images will be resized to 150x150

        batch_size=batch_size,

        class_mode='categorical')  # more than two classes



# this is a similar generator, for validation data

validation_generator = test_datagen.flow_from_directory(

        'data/valid',

        target_size=(240, 240),

        batch_size=batch_size,

        class_mode='categorical')
from keras.preprocessing import image

fnames = [os.path.join('data/train/dandelion', fname) for

fname in os.listdir('data/train/dandelion')]

img_path = fnames[22]

img = image.load_img(img_path, target_size=(240, 240))



x = image.img_to_array(img)

x = x.reshape((1,) + x.shape)

i = 0

f, axes = plt.subplots(1,4,figsize=(14,4))

for batch in train_datagen.flow(x, batch_size=1):

    imgplot = axes[i].imshow(image.array_to_img(batch[0]))

    i += 1

    if i % 4 == 0:

        break

plt.show()
history = model.fit_generator(

          train_generator,

          steps_per_epoch=100,

          epochs=100,

          validation_data=validation_generator,

          validation_steps=50)
model.save('flowers_recognition_v2.h5')
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)



f, axes = plt.subplots(1,2,figsize=(14,4))



axes[0].plot(epochs, acc, 'bo', label='Training acc')

axes[0].plot(epochs, val_acc, 'b', label='Validation acc')

axes[0].legend()



axes[1].plot(epochs, loss, 'bo', label='Training loss')

axes[1].plot(epochs, val_loss, 'b', label='Validation loss')

axes[1].yaxis.set_label_position("right")

axes[1].legend()



plt.show()
# deleting train and test sets, because kaggle is trying to show all

# images that we created as output

shutil.rmtree("/kaggle/working/data")