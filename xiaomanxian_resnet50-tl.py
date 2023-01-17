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
#训练参数设置  （config.py）    

epochs =5

batch_size =168 #当Batch_Size太小，而类别数又比较多的时候,会导致loss函数震荡而不收敛

learn_rate = 0.005

classes = 80

img_width=224

img_height=224



INPUT_SIZE = 224  #输入样本的维度大小

train_num =53879  #训练集样本数

val_num = 7120     #验证集样本数



train_data = "../input/mydataset/train/"

train_annotation_file = "../input/mydataset/json/scene_train_annotations_20170904.json"



val_data = "../input/mydataset/val/"

val_annotation_file= "../input/mydataset/json/scene_validation_annotations_20170908.json"

       

#ResNet50V2_TL

def ResNet50V2_TL(input_shape=(224,224,3),classes=80):

    #实例化不含分类层的ResNet50V2预训练模型，由include_ top=False指定，

    #该模型是用ImageNet数据集训练的，由weights= imagenet ‘指定

    #实例化时会自动下载预训练权重

    path="./resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5"

    

    

    model_base = applications.ResNet50V2(weights='imagenet',include_top=False,

                            input_shape=input_shape)

    

    

    #我们只训练顶部的几层(分类层)

    #锁住所有ResNet50V2的卷积层

    for layer in model_base.layers[:25]:

        layer.trainable = False  #训练期间不会更新层的权重

    #新建一个顺序网络模型

    model=Sequential()

    #添加ResNet50V2预训练模型

    model.add(model_base)

    #添加我们自己的分类层

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dense(128, activation='relu'))

    model.add(Dense(classes, activation='softmax'))

    model.summary()

    return model
# #ResNet50V2_TL

# def ResNet50V2_TL(input_shape=(224,224,3),classes=80):

#     #实例化不含分类层的ResNet50V2预训练模型，由include_ top=False指定，

#     #该模型是用ImageNet数据集训练的，由weights= imagenet ‘指定

#     #实例化时会自动下载预训练权重

#     path="./resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5"

    

#     conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)



#     model = models.Sequential()

#     model.add(conv_base)

#     model.add(layers.Flatten())

#     model.add(Dense(classes, activation='softmax'))

#     return model

#     #conv_base.trainable = False



#     #model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
# # build the VGG16_tl network

# def vgg16_TL(input_shape=(img_width,img_height,3), classes=80):



#     #实例化不含分类层的VGG16预训练模型，由include_top=False指定，

#     #该模型是用ImageNet数据集训练的，由weights= " imagenet '指定

#     # 实例化时会自动下载预训练权重



#     model_base = applications.VGG16(weights='imagenet',

#                                     include_top=False,

#                                     input_shape=input_shape)

# #     plot_model(model_base, to_file='vgg16.png', show_shapes=True)

#     for layer in model_base.layers[:10]:   #[:15]锁住预训练模型的前15层

#         layer.trainable = False  #训练期间不会更新层的权重

#     model_base.summary()

#     model = Sequential()

#     #添加VGG16预训练模型

#     model.add(model_base)

#     #添加我们自己的分类层

#     model.add(GlobalAveragePooling2D())

# #     model.add(Flatten())



# #     model.add(Dense(1024, activation='relu'))  #4096

# #     model.add(Dropout(0.5))



# #     model.add(Dense(1024, activation='relu'))  #4096

# #     model.add(Dropout(0.5))



#     model.add(Dense(classes, activation='softmax'))  #1000

    

# #     model.summary()

    

#     return model
#model.summary()




batch_feature = []



#数据生成器

def data_generator(img_paths, labels, batch_size, is_shuffle=True):

    if is_shuffle:

        img_paths, labels = shuffle(img_paths, labels)

    num_sample = len(img_paths)

    print(num_sample)

    while True:

        if is_shuffle:

            img_paths, labels = shuffle(img_paths, labels)



        for offset in range(0, num_sample, batch_size):

            batch_paths = img_paths[offset:offset + batch_size]

            batch_labels = labels[offset:offset + batch_size]

            batch_labels = np.array(batch_labels)



            batch_features = [load_feature(path) for path in batch_paths]

            batch_labels = np_utils.to_categorical(batch_labels, num_classes=80)



            batch_feature = np.array(batch_features)

            yield batch_feature, batch_labels





# 初始化sess,或回复保存的sess

def start_or_restore_training(sess, saver, checkpoint_dir):

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    if ckpt and ckpt.model_checkpoint_path:

        print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)

        # Restores from checkpoint

        saver.restore(sess, ckpt.model_checkpoint_path)

        step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])

    else:

        sess.run(tf.global_variables_initializer())

        step = 1

        print('start training from new state')

    return sess, step





# 含中文路径读取图片方法

def cv_imread(filepath):

    img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)

    return img





# 等比例缩放图片,size为最边短边长

def resize_img(img, size):

    h = img.shape[0]

    w = img.shape[1]

    scale = max(size / h, size / w)

    resized_img = cv2.resize(img, (int(h * scale), int(w * scale)))

    return resized_img





# 对缩放图片进行随机切割,要求输入图片其中一边与切割大小相等

def random_crop(img, size):

    h = img.shape[0]

    w = img.shape[1]

    if h > w and h > size:

        offset = random.randint(0, h - size)

        croped_img = img[offset:offset + size, :]

    elif h < w and w > size:

        offset = random.randint(0, w - size)

        croped_img = img[:, offset:offset + size]

    elif h == w and h > size:

        offset = random.randint(0, h - size)

        croped_img = img[offset:offset + size, offset:offset + size]

    else:

        croped_img = img

    return croped_img



def load_feature(img_path):

    img = cv_imread(img_path)

    norm_img = img / 255.

    crop = resize_img(norm_img, INPUT_SIZE)

    crop = random_crop(crop, INPUT_SIZE)  # 对缩放图片进行随机切割

    crop = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))

    return crop



def process_annotation(anno_file, dir):

    with open(anno_file) as file:

        annotations = json.load(file)

        img_paths = []

        labels = []

        for anno in annotations:

            img_paths.append(dir + anno["image_id"])

            labels.append(anno["label_id"])

    return img_paths, labels



def make_one_hot(data1):

    return (np.arange(80) == data1[:, None]).astype(np.integer)



def validation(sess, acc, x, y, rate, anno_file, dir, batch_size):

    img_paths, labels = process_annotation(anno_file, dir)

    data_gen = data_generator(img_paths, labels, batch_size, is_shuffle=False)

    num_sample = len(img_paths)

    num_it = ceil(num_sample / batch_size)



    total_accuracy = 0



    for i in range(num_it):

        features, labels = next(data_gen)

        accuracy = sess.run(acc, feed_dict={x: features, y: labels, rate: 1.0})

        total_accuracy += accuracy



    return total_accuracy / num_it
#传入标注文件和图像地址，并返回对应图像地址和图像所属类别标签

trian_img_paths, train_labels = process_annotation(train_annotation_file, train_data )

val_img_paths1, val_labels1 = process_annotation(val_annotation_file, val_data)

#编译模型

model = ResNet50V2_TL((224,224,3), classes=classes)

model.load_weights("../input/output/ResNet50_tl.h5")

sgd = SGD(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=["accuracy"])

#plot_model(model, to_file='model.png', show_shapes=True)

model_name= 'model_best.h5'

checkpoint = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True)#保存最好的模型，模型包含权重和模型

tic = time.time()

# 训练模型

history = model.fit_generator(data_generator(trian_img_paths, train_labels, batch_size),

                              steps_per_epoch=train_num // batch_size,

                              epochs=epochs,

                              validation_data=data_generator(val_img_paths1, val_labels1,batch_size),

                              validation_steps=val_num // batch_size,

                              callbacks=[checkpoint],

                              shuffle=True,

                              verbose=1)  #, learning_rate_reduction



toc = time.time()

print("")

print('used time:',toc - tic,'\n')   #可以输出从开始训练到结束所花费的时间 单位：秒
#保存模型

print("******保存模型******")

model.save_weights('ResNet50_tl.h5')#保存最后一次的权重

123456
class HistoryGraph:

    def __init__(self, history, epochs, title, file_path):

        self.history = history

        self.epochs = epochs

        self.title = title

        self.file_path = file_path

        

    def draw(self):

        figure,(ax1,ax2) = plt.subplots(1, 2, figsize=(12,4))

        figure.suptitle(self.title,fontsize=12)

        figure.subplots_adjust(top=0.85,wspace=0.3)



        epoch_list = list(range(1, self.epochs +1))

        ax1.plot(epoch_list,

                 self.history.history['accuracy'],

                 label='Train Accuracy')

        ax1.plot(epoch_list,

                 self.history.history['val_accuracy'],

                 label='Validation Accuracy')

        ax1.set_xticks(np.arange(0, self.epochs + 1, 5))

        ax1.set_ylabel('Accuracy Value')

        ax1.set_xlabel('Epoch#')

        ax1.set_title('Accuracy')

        ax1.legend(loc="best")



        ax2.plot(epoch_list,

                 self.history.history['loss'],

                 label='Train Loss')

        ax2.plot(epoch_list,

                 self.history.history['val_loss'],

                 label='Validation Loss')

        ax2.set_xticks(np.arange(0, self.epochs + 1, 5))

        ax2.set_ylabel('Loss Value')

        ax2.set_xlabel('Epoch#')

        ax2.set_title('Loss')

        ax2.legend(loc="best")



        plt.savefig(self.file_path)

        plt.close()

from config import *

from util import *

from ResNet50V2_TL import *

from tensorflow.keras.models import load_model





# 测试参数设置

batch_size= 1





#加载训练最好的一次的模型



print("-------加载模型-------")

#model = ResNet50V2_TL((224,224,3), classes=classes)

model_file = "E:/4/al/resnet50z/resnet50v2_tl/model-zuihao.h5"

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

#保存训练性能图

title = "Scene recognition ResNet50V2 Training Performance"

graph_file = "ResNet50V2_Training_Performance.png"

HistoryGraph(history, epochs, title, graph_file).draw()