import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob

import cv2



from scipy.misc import imread

import os

import datetime, time

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from subprocess import check_output
# Examine the total pictures

def get_number_of_file(my_dir):

    return str(len(os.listdir(my_dir)))



print("# of training files: {}".format(get_number_of_file("../input/train")))

print("# of testing files: {}".format(get_number_of_file("../input/test")))
train_labels = pd.read_csv("../input/train_labels.csv")

train_labels.groupby(['invasive']).size().reset_index(name='counts')
def smpl_visual(path, smpl, dim_y):

    

    smpl_pic = glob(smpl)

    fig = plt.figure(figsize=(20, 14))

    

    for i in range(len(smpl_pic)):

        ax = fig.add_subplot(round(len(smpl_pic)/dim_y), dim_y, i+1)

        plt.title("{}: Height {} Width {} Dim {}".format(smpl_pic[i].strip(path),

                                                         plt.imread(smpl_pic[i]).shape[0],

                                                         plt.imread(smpl_pic[i]).shape[1],

                                                         plt.imread(smpl_pic[i]).shape[2]

                                                        )

                 )

        plt.imshow(plt.imread(smpl_pic[i]))

        

    return smpl_pic



smpl_pic = smpl_visual('..input/train\\', '../input/train/112*.jpg', 4)
def visual_with_transformation (pic):



    for idx in list(range(0, len(pic), 1)):

        ori_smpl = cv2.imread(pic[idx])

        smpl_1_rgb = cv2.cvtColor(cv2.imread(pic[idx]), cv2.COLOR_BGR2RGB)

        smpl_1_lab = cv2.cvtColor(cv2.imread(pic[idx]), cv2.COLOR_BGR2LAB)

        smpl_1_gray =  cv2.cvtColor(cv2.imread(pic[idx]), cv2.COLOR_BGR2GRAY) 



        f, ax = plt.subplots(1, 4,figsize=(30,20))

        (ax1, ax2, ax3, ax4) = ax.flatten()

        train_idx = int(pic[idx].strip("../input/train\\").strip(".jpg"))

        print("The Image name: {} Is Invasive?: {}".format(pic[idx].strip("train\\"), 

                                                           train_labels.loc[train_labels.name.values == train_idx].invasive.values)

             )

        ax1.set_title("Original - BGR")

        ax1.imshow(ori_smpl)

        ax2.set_title("Transformed - RGB")

        ax2.imshow(smpl_1_rgb)

        ax3.set_title("Transformed - LAB")

        ax3.imshow(smpl_1_lab)

        ax4.set_title("Transformed - GRAY")

        ax4.imshow(smpl_1_gray)

        plt.show()



visual_with_transformation(smpl_pic)
import keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Sequential

from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.layers import Flatten, Dense, Dropout

from keras.optimizers import SGD

from skimage import io, transform
# Initialize values -



x_train = np.empty(shape=(100, 150, 150, 3))

y_train = np.array(train_labels.invasive.values[0:100])

x_val = np.empty(shape=(100, 150, 150, 3))

y_val = np.array(train_labels.invasive.values[100:200])



for i in range(100):

    tr_img = cv2.imread("../input/train/" + str(i+1) + '.jpg')

    x_train[i] = transform.resize(tr_img, output_shape=(150, 150, 3), mode='constant')



    

for i in range(100):

    val_img = cv2.imread("../input/train/" + str(i+1001) + '.jpg')

    x_val[i] = transform.resize(val_img, output_shape=(150, 150, 3), mode='constant')
# Start some model

model = Sequential()



model.add(ZeroPadding2D((1, 1), input_shape=(150, 150, 3)))



model.add(Convolution2D(64, (3, 3), activation='relu'))

model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D((2, 2), strides=(2, 2)))



model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(128, (3, 3), activation='relu'))

model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D((2, 2), strides=(2, 2)))



model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(256, (3, 3), activation='relu'))

model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(256, (3, 3), activation='relu'))

model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(256, (3, 3), activation='relu'))

model.add(MaxPooling2D((2, 2), strides=(2, 2)))



model.add(Flatten()) # maps back to 1D feature

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))



model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])



print(model.summary())
model.fit(x_train, y_train, epochs=3, batch_size=10)
acc = model.evaluate(x_val, y_val)[1]

print('Evaluation accuracy:{0}'.format(round(acc, 4)))