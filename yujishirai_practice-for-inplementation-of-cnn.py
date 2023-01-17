import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/Data_Entry_2017.csv')

df.head(10)
# create new columns for each decease

pathology_list = ['Cardiomegaly','Emphysema','Effusion','Hernia','Nodule','Pneumothorax','Atelectasis','Pleural_Thickening','Mass','Edema','Consolidation','Infiltration','Fibrosis','Pneumonia']



for pathology in pathology_list :

    df[pathology] = df['Finding Labels'].apply(lambda x: 1 if pathology in x else 0)
df.head(10)
df['Cardiomegaly'].value_counts()
df_t = df[df.Cardiomegaly == 1]

df_f = df[df.Cardiomegaly == 0].sample(3000)



# check

print(df_t.Cardiomegaly.head(10))

print(df_f.Cardiomegaly.head(10))



# combine the true and false data sets, and shuffle them in advance.

df_CM = pd.concat([df_t, df_f], axis=0).sample(frac=1).reset_index(drop=True)

df_CM.Cardiomegaly.sample(10)
import glob

import random

import cv2



images = []

for filename in glob.iglob('../input/**/*.png', recursive=True):

    images.append(filename)



r = random.sample(images, 3)



# Matplotlib black magic

plt.figure(figsize=(16,16))

plt.subplot(131)

plt.imshow(cv2.imread(r[0]))



plt.subplot(132)

plt.imshow(cv2.imread(r[1]))



plt.subplot(133)

plt.imshow(cv2.imread(r[2]));    
images_CM = []



# making fullpath list of the images for analyzation

for file_name in df_CM['Image Index']:

    for img in images:

        if file_name in img:

            images_CM.append(img)
from skimage.io import imread

from skimage.io import imshow

# 画像のサイズを調べる

print(imread(images_CM[0]).shape)



# 画像のリサイズと正規化

images_2d_list = np.zeros([len(images_CM),128,128])

for i, x in enumerate(images_CM):

    image = imread(x, as_grey=True)[::8,::8]

    images_2d_list[i] = (image - image.min())/(image.max() - image.min())
images_2d_list = images_2d_list.reshape(len(images_CM), 128, 128, 1)

images_2d_list.astype('float32')
from keras.models import Sequential

from keras.optimizers import SGD

from keras.utils.vis_utils import plot_model

from keras.layers import Dropout

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import RMSprop



y = df_CM['Cardiomegaly']



model = Sequential()

# conv1

model.add(Conv2D(32, kernel_size=(6,6),

                activation='relu',

                border_mode='same',

                input_shape=(128,128,1)))

model.add(MaxPooling2D(pool_size=(2, 2)))



# conv2

model.add(Conv2D(64, kernel_size=(2,2),

                 activation='relu',

                 border_mode='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))



# fc

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])



history = model.fit(images_2d_list, y, epochs = 30, batch_size = 40, verbose=1, validation_split=0.20)
def history_plot(history):

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()



    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
history_plot(history)