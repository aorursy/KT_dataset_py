# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('../input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as np

import sklearn

import sys

import tensorflow as tf

import time

from tensorflow import keras

import torch

print(tf.__version__)

print(sys.version_info)

for module in mpl,np,pd,sklearn,tf,keras,torch:

    print(module.__name__,module.__version__)
#数据路径

train_dir = "../input/10-monkey-species/training/training"

valid_dir = "../input/10-monkey-species/validation/validation"

label_file = "../input/10-monkey-species/monkey_labels.txt"

#判断路径存在

print(os.path.exists(train_dir))

print(os.path.exists(valid_dir))

print(os.path.exists(label_file))



print(os.listdir(train_dir))

print(os.listdir(valid_dir))
#读取数据

labels = pd.read_csv(label_file,header=0)

print(labels)
#读图

height = 224

width = 224

channels = 3

batch_size = 24

num_classes = 10



train_datagen = keras.preprocessing.image.ImageDataGenerator(

    #对resnet50 预处理图像函数 （针对于tensorflow，函数做了归一化，-1~1）

    preprocessing_function = keras.applications.resnet50.preprocess_input,

 #   rescale = 1./255,     #缩放

    rotation_range = 40,

    width_shift_range = 0.2,

    height_shift_range  = 0.2,

    shear_range = 0.2,

    zoom_range = 0.2,

    horizontal_flip = True,

    fill_mode = 'nearest',

)



#定义 训练集的generator

train_generator = train_datagen.flow_from_directory(train_dir,

                                                   target_size = (height,width),

                                                   batch_size = batch_size,

                                                   seed = 7,

                                                   shuffle = True,

                                                   class_mode = "categorical")



#定义 验证集的generator

valid_datagen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function = keras.applications.resnet50.preprocess_input)

valid_generator = valid_datagen.flow_from_directory(valid_dir,

                                                   target_size = (height,width),

                                                   batch_size = batch_size,

                                                   seed = 7,

                                                   shuffle = False,

                                                    class_mode = "categorical"

                                                   )



#训练集和验证集数目

train_num = train_generator.samples

valid_num = valid_generator.samples

print(train_num,valid_num)
for i in range(2):

    x,y = train_generator.next()

    print(x.shape,y.shape)

    print(y)
resnet50_fine_tune = keras.models.Sequential()

resnet50_fine_tune.add(keras.applications.ResNet50(include_top = False, #去掉最后一层

                                                  pooling = 'avg',

                                                  weights = 'imagenet'))

resnet50_fine_tune.add(keras.layers.Dense(num_classes, activation = 'softmax'))



#预训练网络的第一层次 不训练

resnet50_fine_tune.layers[0].trainable = False



resnet50_fine_tune.compile(loss = "categorical_crossentropy",

             optimizer = "sgd",metrics = ['accuracy'])



resnet50_fine_tune.summary()
#模型的层数   1. 预训练层 和 2.全连接层

len(resnet50_fine_tune.layers)
epochs = 10 

history = resnet50_fine_tune.fit_generator(train_generator,

                              steps_per_epoch = train_num // batch_size,

                             epochs = epochs,

                             validation_data = valid_generator,

                             validation_steps = valid_num // batch_size)
#查看history key

print(history.history.keys())
#打印训练曲线

def plot_learning_curves(history,label,epochs,min_value,max_value):

    data = {}

    data[label] = history.history[label]

    data['val_'+label] = history.history['val_'+label]

    pd.DataFrame(data).plot(figsize=(8,5))

    plt.grid(True)

    plt.axis([0, epochs,min_value,max_value])

    plt.show()

    

plot_learning_curves(history, 'accuracy', epochs, 0, 1)

plot_learning_curves(history, 'loss', epochs, 0, 2)
#上面的实现仅有一层可训练

#另外一种实现 可以训练多层

resnet50 = keras.applications.ResNet50(include_top = False,

                                      pooling = 'avg',

                                      weights = 'imagenet')



resnet50.summary()
#最后6层是可以训练的 (-6表示倒数的第六层)

for layer in resnet50.layers[0:-6]:

    print(layer.trainable)

    layer.trainable = False

    

#然后用这个 resnet50和dense 组合成新的网络

resnet50_new = keras.models.Sequential([

    resnet50,

    keras.layers.Dense(num_classes,activation = 'softmax'),

])



resnet50_new.compile(loss = "categorical_crossentropy",

             optimizer = "sgd",metrics = ['accuracy'])



resnet50_new.summary()
epochs = 10 

history = resnet50_fine_tune.fit_generator(train_generator,

                                           steps_per_epoch = train_num // batch_size,

                                           epochs = epochs,

                                           validation_data = valid_generator,

                                           validation_steps = valid_num // batch_size)
#打印训练曲线

def plot_learning_curves(history,label,epochs,min_value,max_value):

    data = {}

    data[label] = history.history[label]

    data['val_'+label] = history.history['val_'+label]

    pd.DataFrame(data).plot(figsize=(8,5))

    plt.grid(True)

    plt.axis([0, epochs,min_value,max_value])

    plt.show()

    

plot_learning_curves(history, 'accuracy', epochs, 0, 1)

plot_learning_curves(history, 'loss', epochs, 0, 2)