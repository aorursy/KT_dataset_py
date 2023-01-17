# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os,shutil

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

os.listdir('../input/train/train')
original_dataset_dir = '../input/train/train'

base_dir = '../data'

if not os.path.isdir(base_dir): os.mkdir(base_dir)
train_dir = os.path.join(base_dir,'train')

if not os.path.isdir(train_dir): os.mkdir(train_dir)

test_dir = os.path.join(base_dir,'test')

if not os.path.isdir(test_dir): os.mkdir(test_dir)

validation_dir = os.path.join(base_dir,'validation')

if not os.path.isdir(validation_dir): os.mkdir(validation_dir)
train_cats_dir = os.path.join(train_dir,'cats')

if not os.path.isdir(train_cats_dir): os.mkdir(train_cats_dir)



train_dogs_dir = os.path.join(train_dir,'dogs')

if not os.path.isdir(train_dogs_dir): os.mkdir(train_dogs_dir)

    

validation_cats_dir = os.path.join(validation_dir,'cats')

if not os.path.isdir(validation_cats_dir) : os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir,'dogs')

if not os.path.isdir(validation_dogs_dir) : os.mkdir(validation_dogs_dir)



test_cats_dir = os.path.join(test_dir,'cats')

if not os.path.isdir(test_cats_dir) : os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir,'dogs')

if not os.path.isdir(test_dogs_dir) : os.mkdir(test_dogs_dir)
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(train_cats_dir,fname)

    shutil.copyfile(src,dst)   

fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(validation_cats_dir,fname)

    shutil.copyfile(src,dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(test_cats_dir,fname)

    shutil.copyfile(src,dst)



fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(train_dogs_dir,fname)

    shutil.copyfile(src,dst)   

fnames = ['dog.{}.jpg'.format(i) for i in range(1000,1500)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(validation_dogs_dir,fname)

    shutil.copyfile(src,dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500,2000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(test_dogs_dir,fname)

    shutil.copyfile(src,dst)
def showdir(path, depth):

    if depth == 0:

        print("root:[" + path + "]")

 

    for item in os.listdir(path):

        if '.git' not in item:

            print("|      " * depth + "|--" + item)

 

            newitem = os.path.join(path,item)

            if os.path.isdir(newitem):

                showdir(newitem, depth +1)

showdir(base_dir,0)
import pandas as pd

cats = pd.Series([len(os.listdir(train_cats_dir)),len(os.listdir(validation_cats_dir)),len(os.listdir(test_cats_dir))])

dogs = pd.Series([len(os.listdir(train_dogs_dir)),len(os.listdir(validation_dogs_dir)),len(os.listdir(test_dogs_dir))])

df = pd.DataFrame({'Cats':cats,'Dogs':dogs})

df.index = ['Train','Validation','Test']

df
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense

from keras import models
model = models.Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(128,(3,3),activation='relu'))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(128,(3,3),activation='relu'))

model.add(MaxPool2D((2,2)))

model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dense(1,activation='sigmoid'))



model.summary()
from keras import optimizers

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(150,150),batch_size=20,class_mode='binary')

history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=30,validation_data=validation_generator,validation_steps=50)

model.save('cats_and_dogs_small_1.h5')
import matplotlib.pyplot as plt

def show(history,acc,val_acc,label):

    epochs = range(1,31)

    plt.plot(epochs,history.history[acc],label='Training '+label)

    plt.plot(epochs,history.history[val_acc],label='Validation '+label)

    plt.title('Training and Validation '+label)

    plt.legend()

plt.figure(figsize=(15,5))

plt.subplot(121)

show(history,'acc','val_acc','acc')

plt.subplot(122)

show(history,'loss','val_loss','loss')

plt.show()
datagen = ImageDataGenerator(rotation_range=40, 

                             width_shift_range=0.2, 

                             height_shift_range=0.2, 

                             shear_range=0.2,

                             zoom_range=0.2,

                             horizontal_flip=True,

                             fill_mode='nearest')
from keras.preprocessing import image



fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

img_path = fnames[2] #任選一個

img = image.load_img(img_path, target_size=(150,150))

x = image.img_to_array(img)

x = x.reshape((1,)+x.shape)



i = 0

plt.figure(figsize=(8,8))

for batch in datagen.flow(x, batch_size=1):

    plt.subplot(221+i)

    imgplot = plt.imshow(image.array_to_img(batch[0]))

    i +=1

    if i%4 ==0:

        break

plt.show()
from keras.layers import Dropout

model_2 = models.Sequential()

model_2.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))

model_2.add(MaxPool2D(2,2))

model_2.add(Conv2D(64,(3,3),activation='relu'))

model_2.add(MaxPool2D(2,2))

model_2.add(Conv2D(128,(3,3),activation='relu'))

model_2.add(MaxPool2D(2,2))

model_2.add(Flatten())

model_2.add(Dropout(0.5))

model_2.add(Dense(512,activation='relu'))

model_2.add(Dense(1,activation='sigmoid'))



model_2.summary()
model_2.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc']) #0.0001
train_datagen = ImageDataGenerator(rescale=1./255,

                                   rotation_range=40,

                                  width_shift_range=0.2,

                                  height_shift_range=0.2,

                                  shear_range=0.2,

                                  zoom_range=0.2,

                                  horizontal_flip=True,

                                  fill_mode='nearest')



test_datagen = ImageDataGenerator(rescale=1./255) #驗證資料不應該擴充

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150,150),

                                                    batch_size=32,

                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,

                                                        target_size=(150,150),

                                                        batch_size=32,

                                                        class_mode='binary')
history_2 = model_2.fit_generator(train_generator,

                               steps_per_epoch=100,

                               epochs=100,

                               validation_data=validation_generator,

                               validation_steps=50)

model_2.save('cats_and_dogs_small_2.h5')
def show_100(history,acc,val_acc,label):

    epochs = range(1,100+1)

    plt.plot(epochs,history.history[acc],label='Training '+label)

    plt.plot(epochs,history.history[val_acc],label='Validation '+label)

    plt.title('Training and Validation '+label)

    plt.legend()

plt.figure(figsize=(15,5))

plt.subplot(121)

show_100(history_2,'acc','val_acc','acc')

plt.subplot(122)

show_100(history_2,'loss','val_loss','loss')

plt.show()