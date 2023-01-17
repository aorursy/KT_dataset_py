import json

import cv2

import random

import numpy as np 

import tensorflow as tf

import time

import os

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.utils import shuffle

from math import ceil

from tensorflow.keras.optimizers import SGD

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization

from tensorflow.keras.layers import Flatten, Dense  

from tensorflow_core.python.keras.utils import np_utils



from tensorflow.keras.preprocessing.image import ImageDataGenerator



from tensorflow.keras.layers import BatchNormalization



from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Activation



from tensorflow.keras.layers import Input

from tensorflow.keras.models import Model

from tensorflow.keras.layers import add

from tensorflow.keras import backend as K



from tensorflow.keras import applications

from tensorflow.keras.models import load_model

from keras.applications.resnet50 import ResNet50, preprocess_input

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.utils import plot_model

from keras import layers, optimizers, models
import os



from keras import layers, optimizers, models

from keras.applications.resnet50 import ResNet50

from keras.layers import *    

from keras.models import Model

#训练参数设置  （config.py）    

epochs =10

batch_size =128 #当Batch_Size太小，而类别数又比较多的时候,会导致loss函数震荡而不收敛

learn_rate = 0.001

classes = 80

img_width=224

img_height=224



INPUT_SIZE = 224  #输入样本的维度大小

train_num =53879  #训练集样本数

val_num = 7120     #验证集样本数



train_data = "../input/datafl/data/trainimg/"

train_annotation_file = "../input/mydataset/json/scene_train_annotations_20170904.json"



val_data = "../input/datafl/data/valimg/"

val_annotation_file= "../input/mydataset/json/scene_validation_annotations_20170908.json"

       

conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))



model = models.Sequential()

model.add(conv_base)

model.add(layers.Flatten())    #Flatten()多维变一维

model.add(layers.Dense(classes, activation='softmax'))





conv_base.trainable = False



#model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc']) #损失，优化，评估
model.summary()
#定义数据





train_datagen = ImageDataGenerator(

    #preprocessing_function=applications.resnet50.preprocess_input,#后面加的

    rotation_range=40,

    width_shift_range=0.2,    #水平偏移幅度

    height_shift_range=0.2,   #竖直偏移幅度

    shear_range=0.2,          #剪切强度

    zoom_range=0.2,

    horizontal_flip=True, )  #进行随机水平翻







test_datagen = ImageDataGenerator()





train_generator = train_datagen.flow_from_directory(

        # This is the target directory

        train_data,

        # All images will be resized to 150x150

        target_size=(224, 224),

        batch_size=batch_size,

        # Since we use binary_crossentropy loss, we need binary labels

       )



validation_generator = test_datagen.flow_from_directory(

        val_data ,

        target_size=(224, 224),

        batch_size=batch_size,

        )
model=load_model("../input/soutput/ResNet50_tl.h5")
model_name= 'model_best.h5'

checkpoint = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True)#保存最好的模型，模型包含权重和模型

#训练

history = model.fit_generator(

      train_generator,

      steps_per_epoch=train_generator.samples//batch_size,

      epochs=epochs,

      validation_data=validation_generator,

      validation_steps=validation_generator.samples//batch_size ,

      callbacks=[checkpoint])

#保存模型

print("******保存模型******")

model.save('ResNet50_tl.h5',overwrite=True)#保存最后一次的权重

%matplotlib inline



import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



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


from tensorflow.keras.models import load_model





# 测试参数设置







#加载训练最好的一次的模型



print("-------加载模型-------")

#model = ResNet50V2_TL((224,224,3), classes=classes)

model_file = "../input/soutput/ResNet50_tl.h5"

model=load_model(model_file)

#model.load_weights(model_file)



#测试集数据生成器

test_img_paths, test_labels = process_annotation(test_annotation_file, test_data )





#评估网络模型

evaluate = model.evaluate_generator(data_generator(test_img_paths, test_labels, batch_size),

                                    steps=test_num// batch_size)

print("----对测试集进行评估----")

print("test_loss：{:.4f}%".format(evaluate[0]))

print("test_accuracy：{:.2f}%".format(evaluate[1] * 100))
