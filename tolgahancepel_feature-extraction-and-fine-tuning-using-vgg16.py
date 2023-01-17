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

# plotly

import plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px



import tensorflow as tf

from plotly.graph_objs import *

from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array

from keras import layers

from keras import models

from keras import optimizers
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    import itertools

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    #print(cm)

    

    plt.figure(figsize=(8,8))

    plt.grid(False)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()
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

    #perm = np.random.permutation(samples)

    # Copy first 100 samples to the validation directory and rest to the train directory

    for i in range(100):

        name = samples[i].split('/')[-1]

        shutil.copyfile(samples[i],'./data/valid/' + str(category) + '/'+ name)

    for i in range(100,len(samples)):

        name = samples[i].split('/')[-1]

        shutil.copyfile(samples[i],'./data/train/' + str(category) + '/' + name)
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',

                  include_top=False,

                  input_shape=(240, 240, 3))
conv_base.summary()
base_dir = '/kaggle/working/data'

train_dir = os.path.join(base_dir, 'train')

validation_dir = os.path.join(base_dir, 'valid')
datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32
def extract_features(directory, sample_count):

    features = np.zeros(shape=(sample_count, 7, 7, 512))

    labels = np.zeros(shape=(sample_count, 5))



    generator = ImageDataGenerator(rescale=1./255).flow_from_directory(directory,

        target_size=(240, 240),

        batch_size = batch_size, 

        class_mode='categorical')



    i = 0



    print('Entering for loop...');



    

    for inputs_batch, labels_batch in generator:

        features_batch = conv_base.predict(inputs_batch)

        features[i * batch_size : (i + 1) * batch_size] = features_batch

        labels[i * batch_size : (i + 1) * batch_size] = labels_batch

        i += 1

        if i * batch_size >= sample_count:

            break

    return features, labels
train_features, train_labels = extract_features(train_dir, 3823)

validation_features, validation_labels = extract_features(validation_dir, 500)
train_features = np.reshape(train_features, (3823, 7 * 7 * 512))

validation_features = np.reshape(validation_features, (500, 7 * 7 * 512))
model = models.Sequential()

model.add(layers.Dense(2048, activation='relu', input_dim=7 * 7 * 512))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu', input_dim=7 * 7 * 512))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(128, activation='relu', input_dim=7 * 7 * 512))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(5, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
history = model.fit(train_features, train_labels,

                    epochs=25,

                    batch_size=16,

                    validation_data=(validation_features, validation_labels))
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
model_1_val = val_acc[-1]

print("Validation Accuracy: ", model_1_val)
y_pred=model.predict_classes(validation_features)

con_mat = tf.math.confusion_matrix(validation_labels.argmax(1), y_pred)

con_mat = np.array(con_mat)

plot_confusion_matrix(cm = con_mat, classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'], normalize = False)
model = models.Sequential()

model.add(conv_base)

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dropout(0.3))

model.add(layers.Dense(5, activation='softmax'))
model.summary()
train_datagen = ImageDataGenerator(

        rescale=1./255,

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        'data/train',

        target_size=(240, 240),  # all images will be resized to 240x240

        batch_size=batch_size,

        class_mode='categorical')  # more than two classes



validation_generator = test_datagen.flow_from_directory(

        'data/valid',

        target_size=(240, 240),

        batch_size=batch_size,

        class_mode='categorical',

        shuffle = False

)
model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.Adam(lr=2e-5),

              metrics=['acc'])
history = model.fit_generator(

          train_generator,

          epochs=30,

          validation_data=validation_generator)
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
model_2_val = val_acc[-1]

print("Validation Accuracy: ", model_2_val)
validation_generator.reset()

y_pred = model.predict_generator(validation_generator)

y_pred = y_pred.argmax(-1)

con_mat = tf.math.confusion_matrix(validation_generator.classes, y_pred)

con_mat = np.array(con_mat)

plot_confusion_matrix(cm = con_mat, classes = validation_generator.class_indices.keys(), normalize = False)
conv_base.summary()
conv_base.trainable = True



set_trainable = False

for layer in conv_base.layers:

    if layer.name == 'block5_conv1':

        set_trainable = True

    if set_trainable:

        layer.trainable = True

    else:

        layer.trainable = False
model.compile(loss='binary_crossentropy',

              optimizer=optimizers.Adam(lr=2e-5),

              metrics=['acc'])
history = model.fit_generator(

    train_generator,

    steps_per_epoch=100,

    epochs=6,

    validation_data=validation_generator,

    validation_steps=50)
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
model_3_val = val_acc[-1]

print("Validation Accuracy: ", model_3_val)
validation_generator.reset()

y_pred = model.predict_generator(validation_generator)

y_pred = y_pred.argmax(-1)

con_mat = tf.math.confusion_matrix(validation_generator.classes, y_pred)

con_mat = np.array(con_mat)

plot_confusion_matrix(cm = con_mat, classes = validation_generator.class_indices.keys(), normalize = False)
# deleting training and test sets, because kaggle is trying to show all

# images that we created as output

shutil.rmtree("/kaggle/working/data")
my_color = ['Gold','MediumTurquoise','LightGreen']

trace=go.Bar(

            x=['ConvNet from Scratch', 'Feature Extraction w/o DA', 'Feature Extraction w/ DA', 'Fine-Tuning'],

            y=[0.7937, round(model_1_val,4), round(model_2_val,4), round(model_3_val,4)],

            text=[0.7937, round(model_1_val,4), round(model_2_val,4), round(model_3_val,4)],

            textposition='auto',

            marker=dict(

                color=px.colors.sequential.deep,

                line=dict(

                color=px.colors.sequential.deep,

                width=0.4),

            ),

            opacity=1)



data = [trace]

layout = go.Layout(title = 'Accuracies of Models',

              xaxis = dict(title = 'Model'),

              yaxis = dict(title = 'Accuracy'))

fig = go.Figure(data = data, layout = layout)

iplot(fig)