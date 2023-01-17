#所有包的导入

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

#训练参数设置  （config.py）    

EPOCHS =10

BATCH_SIZE =64  #当Batch_Size太小，而类别数又比较多的时候,会导致loss函数震荡而不收敛

lEARN_RATE = 0.0001

CLASSES = 80



INPUT_SIZE = 224   #输入样本的维度大小

train_num =53879  #训练集样本数

val_num = 7120     #验证集样本数



TRAIN_DIR = "../input/mydataset/train/"

TRAIN_ANNOTATION_FILE = "../input/mydataset/json/scene_train_annotations_20170904.json"



VAL_DIR = "../input/mydataset/val/"

VAL_ANNOTATION_FILE = "../input/mydataset/json/scene_validation_annotations_20170908.json"

       
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

    for layer in model_base.layers:

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

    return model

model.summary()

#编译模型 （train.py）      

model = ResNet50V2_TL((224,224,3), classes=CLASSES)

sgd = SGD(lr=lEARN_RATE, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=["accuracy"])

#model=load_model("../input/my_model_ResNet.h5")




#utils工具文件 （utils.py）

batch_feature = []

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

    crop = random_crop(crop, INPUT_SIZE)

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

    return (np.arange(80) == data1[:, None]).astype(np.integer)       #80分类

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



trian_img_paths, train_labels = process_annotation(TRAIN_ANNOTATION_FILE, TRAIN_DIR)

trian_img_paths1, train_labels1 = process_annotation(VAL_ANNOTATION_FILE, VAL_DIR)
# 训练模型

tic=time.time()

history = model.fit_generator(data_generator(trian_img_paths, train_labels, BATCH_SIZE),

                              steps_per_epoch=train_num // BATCH_SIZE,

                              epochs=EPOCHS,

                              validation_data=data_generator(trian_img_paths1, train_labels1, BATCH_SIZE),

                              validation_steps=val_num // BATCH_SIZE,

                              shuffle=True,

                              verbose=1)

#max_queue_size缓存batch的，max_queue_size是多少就缓存几个batch

toc = time.time()

print("")

print('used time:',toc - tic,'\n')   #可以输出从开始训练到结束所花费的时间 单位：秒
#保存模型

print("******保存模型******")

model.save_weights('my_model_ResNet.h5')
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


#保存训练性能图

title = "Scene recognition ResNet50V2 Training Performance"

graph_file = "ResNet50V2_Training_Performance.png"

HistoryGraph(history, EPOCHS, title, graph_file).draw()