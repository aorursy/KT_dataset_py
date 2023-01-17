# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from keras.applications import VGG16



conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))
conv_base.summary()
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
import shutil

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
import pandas as pd

cats = pd.Series([len(os.listdir(train_cats_dir)),len(os.listdir(validation_cats_dir)),len(os.listdir(test_cats_dir))])

dogs = pd.Series([len(os.listdir(train_dogs_dir)),len(os.listdir(validation_dogs_dir)),len(os.listdir(test_dogs_dir))])

df = pd.DataFrame({'Cats':cats,'Dogs':dogs})

df.index = ['Train','Validation','Test']

df
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)

batch_size = 20
def extract_feature(directory, sample_cout):

    features = np.zeros(shape=(sample_cout,4,4,512))

    labels = np.zeros(shape=(sample_cout))

    generator = datagen.flow_from_directory(directory,target_size=(150,150),batch_size=batch_size,class_mode='binary')

    i = 0

    

    for inputs_batch, labels_batch in generator:

        features_batch = conv_base.predict(inputs_batch)

        features[i*batch_size:(i+1)*batch_size] = features_batch

        labels[i*batch_size:(i+1)*batch_size]   = labels_batch

        i+=1

        print(i,end='')

        if i*batch_size >=sample_cout:

            break

    return features, labels



train_features, train_labels = extract_feature(train_dir, 2000)

validation_features, validation_labels = extract_feature(validation_dir,1000)

test_features, test_labels = extract_feature(test_dir, 1000)
train_features = np.reshape(train_features,(2000,-1))

validation_features = np.reshape(validation_features,(1000,-1))

test_features = np.reshape(test_features,(1000,-1))
from keras.models import Sequential

from keras.layers import Dense,Dropout

from keras.optimizers import RMSprop



model = Sequential()

model.add(Dense(256,activation='relu',input_dim=4*4*512))

model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))



model.compile(optimizer=RMSprop(lr=2e-5),loss='binary_crossentropy',metrics=['acc'])

history = model.fit(train_features,train_labels,epochs=30,batch_size=20,validation_data=(validation_features,validation_labels))
import matplotlib.pyplot as plt

def show_all(history):

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

show_all(history)
from keras.layers import Flatten

conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))

model2 = Sequential()

model2.add(conv_base)

model2.add(Flatten())

model2.add(Dense(256,activation='relu'))

model2.add(Dense(1,activation='sigmoid'))

model2.summary() 
#conv_base.trainable = False # Very important
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,

                                   rotation_range=40,

                                  width_shift_range=0.2,

                                  height_shift_range=0.2,

                                  shear_range=0.2,

                                  zoom_range=0.2,

                                  horizontal_flip=True,

                                  fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(150,150),batch_size=20,class_mode='binary')



model2.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=2e-5),metrics=['acc'])
history_2 = model2.fit_generator(train_generator,

                                steps_per_epoch=100,

                                epochs=30,

                                validation_data=validation_generator,

                                validation_steps=50)
show_all(history_2)
print('Quick Feature Extraction:cannot Data Augmentation')

show_all(history)

print('Quick Feature Extraction:Data Augmentation')

show_all(history_2)