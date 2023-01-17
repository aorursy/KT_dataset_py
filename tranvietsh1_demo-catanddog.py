# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



import keras

import matplotlib.pyplot as plt



import tensorflow as tf

print(os.listdir("../input/dogandcat/dogandcat/train"))



# Any results you write to the current directory are saved as output.
N_CLASSES = 2

BATCH_SIZE = 64

W = H = 128

classes = ['cat', 'dog']
train_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0)
test_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0)
train_dataset = train_generator.flow_from_directory(directory='../input/dogandcat/dogandcat/train/',

                                                    target_size=(W, H),

                                                   batch_size=BATCH_SIZE,

                                                   class_mode='binary')

print("class indices train_dataset: " + str(train_dataset.class_indices))





test_dataset = test_generator.flow_from_directory(directory='../input/dogandcat/dogandcat/test/',

                                                    target_size=(W, H),

                                                   batch_size=BATCH_SIZE,

                                                   class_mode='binary')

print("class indices test_dataset: " + str(train_dataset.class_indices))
for data_batch, labels_batch in train_dataset:

    print('data batch shape:', data_batch.shape)

    print('labels batch shape:', labels_batch.shape)

    break
def show_demo(train_dataset):

    grids = (4,4) # chia theo khổ hiện thi ảnh , ở đây là 16 ảnh theo khổ 4x4

    counter = 0



    plt.figure(figsize=(10,10)) 



    for batch_images, batch_labels in train_dataset:

        i = np.random.randint(len(batch_images))

        img = batch_images[i]

        label = batch_labels[i]



        if(counter < grids[0]*grids[1]):

            counter += 1

        else:

            break



        # plot image and its label

        ax = plt.subplot(grids[0], grids[1], counter) # tạo khổ cho ax

        ax = plt.imshow(img, cmap='brg') # in ảnh ra và fit vào khổ đã tạo

        plt.xticks([]) # bỏ cái đường viền chỉ số cột ox(bình thường là nó có chỉ số)

        plt.yticks([]) # bỏ cái đường viền chỉ số cột oy(bình thường là nó có chỉ số)

        plt.title(classes[int(label)]) # gán tên trùng với tên label cho ảnh
show_demo(train_dataset)
from keras import layers

from keras import models

def cnn_model():

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(W, H, 3)))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    return model
model = cnn_model()

model.summary()
from keras import optimizers

model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])

history1 = model.fit_generator(train_dataset, epochs=10, steps_per_epoch=30, validation_data=test_dataset, validation_steps=10)
#save model

model.save('cats_and_dogs_small_1.h5')
test_sample_images, test_sample_labels = next(test_dataset)
# make prediction

predict_sample_labels = (model.predict_on_batch(test_sample_images) > 0.5).astype(int)
def test_data(test_sample_images,test_sample_labels,predict_sample_labels):

    grids = (3,3)

    counter = 0



    plt.figure(figsize=(10,10))



    for img, gt_label, predict_label in zip(test_sample_images, test_sample_labels, predict_sample_labels):



        if(counter < grids[0]*grids[1]):

            counter += 1

        else:

            break



        # plot image and its label

        ax = plt.subplot(grids[0], grids[1], counter)

        ax = plt.imshow(img, cmap='brg')

        plt.xticks([])

        plt.yticks([])

        plt.title("Actual: %s    Predict: %s"%(classes[int(gt_label)], classes[int(predict_label)]))
test_data(test_sample_images,test_sample_labels,predict_sample_labels)
import matplotlib.pyplot as plt

def visualize_data(history1):

    acc = history1.history['acc']

    val_acc = history1.history['val_acc']

    loss = history1.history['loss']

    val_loss = history1.history['val_loss']

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
visualize_data(history1)
train_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0,

                                                               rotation_range=40,#xoay ngau nhien quanh 40 do

                                                               #RGB->HSV, tang V len random(0.5%-2%) ->RGB

                                                               brightness_range=(0.5, 2),

                                                               #dich ngau nhien chieu doc(tinh theo %)

                                                               height_shift_range = 0.25,

                                                               #dich ngau nhien chieu ngang(tinh theo %)

                                                               width_shift_range = 0.25,

                                                               zoom_range = 0.5,#phong to anh ngau nhien

                                                               shear_range = 0.5,#cat anh ngau nhien

                                                               horizontal_flip=True#lat anh ngau nhien(theo chieu ngang)

                                                              )
print(os.listdir("../input/dogandcat/dogandcat/train/dog"))
# load path image demo

root_dir = "../input/dogandcat/dogandcat/train/dog"

all_path_img_dog = os.listdir(root_dir)

img_path_demo = root_dir +'/'+ all_path_img_dog[0]

print(img_path_demo)
#load image and show demo a image with Augmentation

from keras.preprocessing import image

img = image.load_img(img_path_demo, target_size=(W, H))

x = image.img_to_array(img)

x = x.reshape((1,) + x.shape)



grids = (3,3)

counter = 0

plt.figure(figsize=(10,10))



for batch in train_generator.flow(x, batch_size=1):

    if(counter < grids[0]*grids[1]):

        counter += 1

    else:

        break

    # plot image and its label

    ax = plt.subplot(grids[0], grids[1], counter) # tạo khổ cho ax

    ax = plt.imshow(image.array_to_img(batch[0]))

plt.show()
# load all image for training and testing

train_dataset = train_generator.flow_from_directory(directory='../input/dogandcat/dogandcat/train/',

                                                    target_size=(W, H),

                                                   batch_size=BATCH_SIZE,

                                                   class_mode='binary')

print(train_dataset.class_indices)



test_dataset = test_generator.flow_from_directory(directory='../input/dogandcat/dogandcat/test/',

                                                    target_size=(W, H),

                                                   batch_size=BATCH_SIZE,

                                                   class_mode='binary')
show_demo(train_dataset)
model = cnn_model()

model.summary()
#Compile and fit

model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])

history_Agu = model.fit_generator(train_dataset, epochs=10, steps_per_epoch=30, validation_data=test_dataset, validation_steps=10)
test_sample_images, test_sample_labels = next(test_dataset)



# make prediction

predict_sample_labels = (model.predict_on_batch(test_sample_images) > 0.5).astype(int)

test_data(test_sample_images,test_sample_labels,predict_sample_labels)
#visualize_data

visualize_data(history_Agu)