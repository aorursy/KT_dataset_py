# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from sklearn.preprocessing import LabelEncoder

from keras.utils.np_utils import to_categorical

import keras







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/dataset/dataset"))



# Any results you write to the current directory are saved as output.
#define number image and size

W=H = 224

number_classes = 17

batch_size=32
#get image name

def getname(img_path):

    return img_path.split('/')[-2]
#read image

root_dir = '../input/dataset/dataset'

all_path_imgs = glob.glob(os.path.join(root_dir, '*/*.jpg'))

print(all_path_imgs[0])

np.random.shuffle(all_path_imgs)



imgs = []

labels = []

for img_path in all_path_imgs:

    img = load_img(img_path, target_size=(W,H))

    img = img_to_array(img)

    imgs.append(img)

    

    name = getname(img_path)

    labels.append(name)

train_data = np.array(imgs)

le = LabelEncoder()

y_encode = le.fit_transform(labels)

train_labels = to_categorical(y_encode)



#plot data 

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20,20))

row = col = 9

for i in range(1, row*col+1):

    k = np.random.randint(1, len(train_data))

    fig.add_subplot(row, col, i)

    img = load_img(all_path_imgs[k], target_size=(80,80))

    plt.imshow(img)

    plt.xticks([])

    plt.yticks([])

    plt.title(labels[k])

plt.show()

#train test split 

from sklearn.model_selection import train_test_split

X_train, X_test , y_train , y_test = train_test_split(train_data, train_labels, test_size = 0.2, random_state = 16)
# augmentation cho training data

aug_train = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, 

                         zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

# augementation cho test

aug_test= keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
#load vgg16

vgg = keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(W,H,3))



vgg_layers = len(vgg.layers)

print('Number vgg layers: '+ str(vgg_layers))



vgg_shape = vgg.output_shape

print('Number vgg_shape: '+ str(vgg_shape))



from keras import models

from keras import layers



#define top model

flatten = layers.Flatten()(vgg.output)



fc = layers.Dense(256, activation='relu')(flatten)

fc = layers.Dropout(0.5)(fc)



fc = layers.Dense(number_classes, activation='softmax')(fc)



#new model 

model = models.Model(vgg.input, fc)
#freeze vgg model 

for layer in vgg.layers:

    layer.trainable=False



opt = keras.optimizers.RMSprop(0.001)

model.compile(optimizer=opt,

              loss='categorical_crossentropy',

              metrics=(['acc']))

network = model.fit_generator(aug_train.flow(X_train, y_train, batch_size=batch_size),

                              steps_per_epoch=len(X_train)//batch_size,

                              validation_data=aug_test.flow(X_test, y_test, batch_size),

                              validation_steps=len(X_test)//batch_size,

                              epochs=25)
#unfreeze vgg model 

for layer in vgg.layers[15:]:

    layer.trainable=True



opt = keras.optimizers.SGD(0.001)

model.compile(optimizer=opt,

              loss='categorical_crossentropy',

              metrics=(['acc']))

network = model.fit_generator(aug_train.flow(X_train, y_train, batch_size=batch_size),

                              steps_per_epoch=len(X_train)//batch_size,

                              validation_data=aug_test.flow(X_test, y_test, batch_size),

                              validation_steps=len(X_test)//batch_size,

                              epochs=35)
#visualize 

history_dict = network.history

history_dict.keys()
epochs = range(1, len(history_dict['acc'])+1)

plt.plot(epochs, history_dict['acc'], 'b', label='acc')

plt.plot(epochs, history_dict['val_acc'], 'r', label='val_acc')

plt.xlabel("epochs")

plt.ylabel("accuracy")

plt.title("Training and validation acc")

plt.legend()

plt.figure()

plt.plot(epochs, history_dict['loss'], 'b', label='loss')

plt.plot(epochs, history_dict['val_loss'], 'r', label='val_loss')

plt.xlabel("epochs")

plt.ylabel("loss")

plt.title("Training and validation loss")

plt.legend()