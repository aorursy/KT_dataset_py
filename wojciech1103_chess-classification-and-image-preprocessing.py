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
import imageio

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from PIL import Image, ImageOps, ImageFilter

import scipy.ndimage as ndi

from sklearn.metrics import classification_report, confusion_matrix



from keras.models import Sequential

from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Activation, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing import image

from keras.utils import plot_model
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
os.mkdir('/kaggle/working/chess')



os.mkdir('/kaggle/working/chess/bishop')

os.mkdir('/kaggle/working/chess/knight')

os.mkdir('/kaggle/working/chess/queen')

os.mkdir('/kaggle/working/chess/rook')

os.mkdir('/kaggle/working/chess/king')

os.mkdir('/kaggle/working/chess/pawn')
dirname_work = '/kaggle'

dir_work = os.path.join('/kaggle', 'working')

dir_work_chess = os.path.join(dir_work, 'chess')





bishop_path_work = os.path.join(dir_work_chess, 'bishop')



knight_path_work = os.path.join(dir_work_chess, 'knight')



queen_path_work = os.path.join(dir_work_chess, 'queen')



rook_path_work = os.path.join(dir_work_chess, 'rook')



king_path_work = os.path.join(dir_work_chess, 'king')



pawn_path_work = os.path.join(dir_work_chess, 'pawn')

dir_chess_folders_work = os.listdir(dir_work_chess)

dir_chess_paths_work = [os.path.join(dir_work_chess, path) for path in dir_chess_folders_work]
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
chess_dic = {}

for path in dir_chess_paths:

    head, tail = os.path.split(path)

    chess_dic[tail] = len(os.listdir(path))
label_list = ["{}: {}".format(key, chess_dic[key]) for key in chess_dic]
plt.figure(figsize=(8, 8))

plt.bar(range(len(chess_dic)), list(chess_dic.values()), color="green")

plt.xticks(range(len(chess_dic)), list(label_list))

plt.show();
def plot_img_hist(item_dir, num_img=6):

    all_item_dirs = os.listdir(item_dir)

    item_files = [os.path.join(item_dir, file) for file in all_item_dirs][:num_img]



    #plt.figure(figsize=(10, 10))

    for idx, img_path in enumerate(item_files):

        fig1 = plt.figure(idx,figsize=(10, 10))

        fig1.add_subplot(2, 2, 1)

        img = mpimg.imread(img_path, 0)

        plt.imshow(img)

        fig1.add_subplot(2, 2, 2)

        plt.hist(img.ravel(),bins=256, fc='k', ec='k')



    plt.tight_layout()
for path in dir_chess_paths:

    plot_img_hist(path, 8)
def image_binarization(path_from, path_to):



    i=1

    files = os.listdir(path_from)

    for file in files: 

        try:

            file_dir = os.path.join(path_from, file)

            file_dir_save = os.path.join(path_to, file)

            img = Image.open(file_dir)

            img = img.convert("1")

            img.save(file_dir_save) 

            i=i+1

        except:

            continue
image_binarization(dir_chess_paths[0], bishop_path_work)
image_binarization(dir_chess_paths[1], king_path_work)
image_binarization(dir_chess_paths[2], rook_path_work)
image_binarization(dir_chess_paths[3], pawn_path_work)
image_binarization(dir_chess_paths[4], queen_path_work)
image_binarization(dir_chess_paths[5], knight_path_work)
for path in dir_chess_paths_work:

    head, tail = os.path.split(path)

    plot_imgs(path, tail, 8, 'binary')
def image_median_filtering(path_from, path_to, window_size=3):



    i=1

    files = os.listdir(path_from)

    for file in files: 

        try:

            file_dir = os.path.join(path_from, file)

            file_dir_save = os.path.join(path_to, file)

            img = Image.open(file_dir)

            img = img.filter(ImageFilter.MedianFilter(window_size))

            img.save(file_dir_save) 

            i=i+1

        except:

            continue
image_median_filtering(bishop_path_work, bishop_path_work)
image_median_filtering(king_path_work, king_path_work)
image_median_filtering(rook_path_work, rook_path_work)
image_median_filtering(pawn_path_work, pawn_path_work)
image_median_filtering(queen_path_work, queen_path_work)
image_median_filtering(knight_path_work, knight_path_work)
for path in dir_chess_paths_work:

    head, tail = os.path.split(path)

    plot_imgs(path, tail, 8, 'binary')
for path in dir_chess_paths_work:

    plot_img_hist(path, 4)
img_size_h = 300

img_size_w = 300
train_datagen = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.3,

    rotation_range=90,

    width_shift_range=0.6,

    height_shift_range=0.6,

    shear_range=3, 

    zoom_range=50,

    horizontal_flip=True,

    vertical_flip=True)
input_shape = (img_size_h, img_size_w, 1) 
batch_size = 16

train_generator = train_datagen.flow_from_directory(

    dir_work_chess,

    target_size=(img_size_h, img_size_w),

    color_mode='grayscale',

    batch_size=batch_size,

    class_mode='categorical',

    subset='training',

    shuffle=True, #we shuffle our images for better performance

    seed=8)



validation_generator = train_datagen.flow_from_directory(

    dir_work_chess,

    target_size=(img_size_h, img_size_w),

    color_mode='grayscale',

    batch_size=batch_size,

    class_mode='categorical',

    subset='validation',

    shuffle=True,

    seed=7)
model = Sequential([



    Conv2D(16, (5,5), input_shape=input_shape, padding='same', activation='relu'),

    Conv2D(32, (3,3), padding='same', activation='relu'),

    Conv2D(32, (3,3), padding='same', activation='relu'),



    Conv2D(32, (3,3), padding='same', activation='relu'),

    MaxPool2D((2,2)),

    BatchNormalization(momentum=0.3),

    Dropout(0.2),

    

    Conv2D(32, (5,5), padding='same', activation='relu'),    

    Conv2D(64, (3,3), padding='same', activation='relu'),

    Conv2D(64, (3,3), padding='same', activation='relu'),



    Conv2D(64, (3,3), padding='same', activation='relu'),

    MaxPool2D((2,2)),

    BatchNormalization(momentum=0.3),

    Dropout(0.2),



    

    Flatten(),

    

    Dense(128, activation='relu'),

    Dropout(0.4),

    Dense(6, activation='softmax')

    

    

])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.0005) 

callback = [learning_rate_reduction]
history = model.fit_generator(

    train_generator,

    epochs=500,

    validation_data=validation_generator,

    callbacks = callback

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

num_of_test_samples = 162 

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