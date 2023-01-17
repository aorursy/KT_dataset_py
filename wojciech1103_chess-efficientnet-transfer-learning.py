# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
! pip install -U efficientnet
# https://github.com/qubvel/efficientnet#installation



import efficientnet.keras as efn 
import imageio

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from PIL import Image, ImageOps, ImageFilter

import scipy.ndimage as ndi

from sklearn.metrics import classification_report, confusion_matrix



from keras.layers import Dense,GlobalAveragePooling2D, MaxPool2D, Dropout



from keras.preprocessing import image

from keras.applications.mobilenet import preprocess_input

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau

from keras.utils import plot_model

from keras import regularizers
jpg_counter = 0

png_counter = 0



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename[-3:] == "jpg":

            jpg_counter = jpg_counter + 1

        elif filename[-3:] == "png":

            png_counter = png_counter + 1



print("Number of jpg: {}\nNumber of png: {}".format(jpg_counter, png_counter))
dirname = '/kaggle/input/chessman-image-dataset/Chessman-image-dataset/Chess'

dir_chess_folders = os.listdir(dirname)

dir_chess_paths = [os.path.join(dirname, path) for path in dir_chess_folders]
dir_chess_paths
def plot_imgs(item_dir, title=" ", num_imgs=4, cmap='viridis'):

    all_item_dirs = os.listdir(item_dir)

    item_files = [os.path.join(item_dir, file) for file in all_item_dirs][:num_imgs]



    plt.figure(figsize=(15, 15))

    for idx, img_path in enumerate(item_files):

        plt.subplot(8, 8, idx+1)

        img = plt.imread(img_path, 0)

        plt.title(title)

        plt.imshow(img, cmap=cmap)



    plt.tight_layout()
for path in dir_chess_paths:

    head, tail = os.path.split(path)

    plot_imgs(path, tail, 8)
img_size_h = 300

img_size_w = 300
train_datagen = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.2,

    rotation_range=45,

    width_shift_range=0.5,

    height_shift_range=0.5,

    shear_range=0.5, 

    zoom_range=10,

    horizontal_flip=True,

    vertical_flip=True)
input_shape = (img_size_h, img_size_w, 3) 
batch_size_train = 64

batch_size_test = 32



train_generator = train_datagen.flow_from_directory(

    dirname,

    target_size=(img_size_h, img_size_w),

    color_mode='rgb',

    batch_size=batch_size_train,

    class_mode='categorical',

    subset='training',

    shuffle=True, #we shuffle our images for better performance

    seed=8)



validation_generator = train_datagen.flow_from_directory(

    dirname,

    target_size=(img_size_h, img_size_w),

    color_mode='rgb',

    batch_size=batch_size_test,

    class_mode='categorical',

    subset='validation',

    shuffle=True,

    seed=7)
base_model = efn.EfficientNetB3(weights='imagenet')
base_model.summary()
base_model.layers.pop()

base_model.layers.pop()
x=base_model.output

x=Dropout(0.3)(x)

x=Dense(1024, activation='relu', kernel_regularizer=regularizers.l1(0.01))(x)

x=Dense(1024, activation='relu', kernel_regularizer=regularizers.l1(0.01))(x)

x=Dense(512, activation='relu', kernel_regularizer=regularizers.l1(0.01))(x)

preds=Dense(6,activation='softmax')(x) #final layer with softmax activation
model=Model(inputs=base_model.input,outputs=preds)

#specify the inputs

#specify the outputs

#now a model has been created based on our architecture
for i,layer in enumerate(model.layers):

    print(i,layer.name)
for layer in model.layers[:329]:

    layer.trainable=False

for layer in model.layers[329:]:

    layer.trainable=True
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.summary()
step_size_train=train_generator.n//train_generator.batch_size
learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.001) 

callback = [learning_rate_reduction]
history = model.fit_generator(

    train_generator,

    epochs=300,

    steps_per_epoch=step_size_train,

    validation_data=validation_generator,

    callbacks=callback

    )
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# https://gist.github.com/RyanAkilos/3808c17f79e77c4117de35aa68447045

num_of_test_samples = 109   

batch_size = 32

#Confution Matrix and Classification Report

Y_pred = model.predict_generator(validation_generator, num_of_test_samples // batch_size+1)

y_pred = np.argmax(Y_pred, axis=1)



matrix1 = confusion_matrix(validation_generator.classes, y_pred)



# Using: https://getaravind.com/blog/confusion-matrix-seaborn-heatmap/



sns.heatmap(matrix1,annot=True,cbar=False);

plt.ylabel('True Label');

plt.xlabel('Predicted Label');

plt.title('Confusion Matrix');
print('\nClassification Report')

target_names = ['Bishop',

                 'King',

                 'Rook',

                 'Pawn',

                 'Queen',

                 'Knight']

class_report = classification_report(validation_generator.classes, y_pred, target_names=target_names)

print(class_report)
!ls
import shutil

shutil.rmtree("/kaggle/working/chess")