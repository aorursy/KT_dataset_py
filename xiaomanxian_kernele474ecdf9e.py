

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



from tensorflow.keras.layers import BatchNormalization



from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Activation



from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau



from tensorflow.keras.layers import Input

from tensorflow.keras.models import Model

from tensorflow.keras.layers import add

from tensorflow.keras import backend as K

import matplotlib.pyplot as plt

%matplotlib inline

from tensorflow.keras.models import load_model

#训练参数设置  （config.py）    #需修改，改成你的参数及路径

EPOCHS =20

BATCH_SIZE =128

lEARN_RATE = 0.0001

CLASSES = 80



INPUT_SIZE = 229   #输入样本的维度大小

train_num =53879  #训练集样本数

val_num = 7120     #验证集样本数



TRAIN_DIR = "../input/mydataset/train/"

TRAIN_ANNOTATION_FILE = "../input/mydataset/json/scene_train_annotations_20170904.json"



VAL_DIR = "../input/mydataset/val/"

VAL_ANNOTATION_FILE = "../input/mydataset/json/scene_validation_annotations_20170908.json"



#build ResNet50V2 模型  （ResNet50V2.py）    

class ResNet50V2:



    @staticmethod

    def residual_module(x,

                        filters,

                        kernel_size=3,

                        stride=2,

                        conv_shortcut=False):

        """

        创建ResNet的参模块

        Args ;

        X:             参差模块的输入

        filters :          瓶颈层核数

        kernel size:      瓶颈层和大小，缺省为3

        stride :         第一卷积层的卷积步长

        conv_shortcut:   True时用卷积短路，False时使用等维短路

        Returns :

        生成的参差模块

        """

        bn_axis=3 if K.image_data_format()=='channels_last' else 1

        preact=BatchNormalization(axis=bn_axis,epsilon=1.001e-5)(x)

        preact =Activation('relu')(preact)

        if conv_shortcut is True:

            shortcut= Conv2D(4 * filters,

                            kernel_size=(1,1),

                            strides=stride,

                            padding='valid',

                            kernel_initializer='he_normal' ) ( preact )

            x = Conv2D(filters=filters,

                        kernel_size=(1, 1),

                        strides=stride,

                        padding='valid',

                        kernel_initializer='he_normal')(preact)

        else:



            #shortcut = Conng2D(1， strides=stride)(X) if stride > 1 else X

            shortcut = x

            x = Conv2D(filters=filters,

                        kernel_size=(1, 1),

                        strides=(1, 1),

                        kernel_initializer='he_normal')(preact)

        #参差模块第(2)卷积块:先归一化，再激励，在卷积

        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)

        x = Activation("relu")(x)

        x = Conv2D(filters=filters,

                    kernel_size=kernel_size,

                    strides=(1,1),

                    padding='same' ,

                    kernel_initializer='he_normal')(x)

        #参差模块第③卷积块:先归一化，再激励，在卷积

        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)

        x = Activation("relu")(x)

        # 1*1卷积

        x = Conv2D(4 * filters,

                    kernel_size=(1, 1),

                    strides=(1,1),

                    padding='valid',

                    kernel_initializer='he_normal')(x)

        #将最后的卷积结果和短路分支块相加

        x = add([shortcut, x])

        #返回参差块

        return x



    @staticmethod

    def stack(x, filters, blocks,stride1=2):

        """

        生成一组堆叠的参差块

        Args:

        X:          参差模块的输入

        filters:       瓶颈层核数

        blocks:       参差块数

        stride1:       缺省为2， 第一块第一层的卷积步长

        Returns:

        一组堆叠的参差块

        """

        # 1、所有残差网段的第一个残差模块均使用前激励卷积残差模块

        # 2、第一个残差网段的第--个残差模块卷积步长为1，所以不降维

        #3、第二、三、四残差网段的第一-个残差模块卷积步长为2，所以产生3次减半降维

        x = ResNet50V2.residual_module(x, filters, stride=stride1, conv_shortcut=True)

         #所有残差网段的非第-个残差模块均使用前激励等维残差模块

        for i in range(1,blocks) :

                x= ResNet50V2.residual_module(x,filters, conv_shortcut=False)

        return x



    @staticmethod

    def build(width,height,depth,classes):

        """

        构造参差网络模型

        Args:

        width:          样本宽度

        height:         样本高度

        depth:          样本通道数

        classes:         类别数量

        Returns :

        参差网络模型

        """

        #初始化输入维度和通道位置 (channels_last,通道后置)

        input_shape = (height, width, depth)

        bn_axis = 3

        #如果我们的样本为通道前(channels_ first)

        if K.image_data_format() =="channels_first":

            input_shape = (depth,height,width)

            bn_axis=1

        #设置网络的输入

        inputs = Input(shape=input_shape)

        x = Conv2D(64, 7, strides=2, padding='same', use_bias=True)(inputs)

        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)

        x = Activation("relu")(x)

        x = MaxPooling2D(3,strides=2, padding='same')(x)

        x = ResNet50V2.stack(x, 64, 3, stride1=1)

        x = ResNet50V2.stack(x, 128, 4, stride1=2)

        x = ResNet50V2.stack(x, 256, 6, stride1=2 )

        x = ResNet50V2.stack(x, 512, 3, stride1=2 )

        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)

        x = Activation("relu")(x)

        #全局平均池化

        x = GlobalAveragePooling2D()(x)

        #全连接

        x = Dense(classes,activation='softmax',name='output')(x)

        #创建模型

        model = Model(inputs,x,name="resnet50v2")

        #返回模型

        return model

 

  
'''

aug=ImageDataGenerator (rotation range=20,



                        zoom_range=0.15,



                        width_shift_range=0.2,

                        height_shift_ range=0.2,

                        shear_range=0.15,

                        horizontal_flip=True,

                        fill_mode= " nearest" )



#加載訓紘集的RGB均値



means=json.loads(open(setting.DATASET_MEAN_PILE).read()) #初始化图像预处理器



sp = SimplePreprocessor(227, 227)



pp = PatchPreprocessor(227, 227)



mp=MeanPreprocessor (means["R"], means["G"], means["B" ])

iap=ImageToArrayPreprocessor()



#初始化训练数据集生成器



trainGen - HDP5DatasetGenerator (setting . TRAIN_ HDF5,



128,

'''


#编译模型 （train.py）      

model=ResNet50V2.build(width=229,height=229,depth=3,classes=CLASSES)

model.load_weights('../input/resnet1/ResNet50V22.h5')

sgd = SGD(lr=lEARN_RATE, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=["accuracy"])



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

    return (np.arange(80) == data1[:, None]).astype(np.integer)       #需修改

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

# 训练模型 （train.py）

model_name= 'model.h5'

tic=time.time()

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1,factor=0.1, min_lr=0.000001)

checkpoint = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True)



trian_img_paths, train_labels = process_annotation(TRAIN_ANNOTATION_FILE, TRAIN_DIR)

trian_img_paths1, train_labels1 = process_annotation(VAL_ANNOTATION_FILE, VAL_DIR)

tic = time.time()

history = model.fit_generator(data_generator(trian_img_paths, train_labels, BATCH_SIZE),

                              steps_per_epoch=train_num // BATCH_SIZE,

                              epochs=EPOCHS,

                              validation_data=data_generator(trian_img_paths1, train_labels1, BATCH_SIZE),

                              validation_steps=val_num // BATCH_SIZE,

                              shuffle=True,

                              verbose=1)

toc = time.time()  

print("")



print('used time:',toc - tic,'\n')   #可以输出从开始训练到结束所花费的时间  单位：秒

#保存模型

print("******保存模型******")

# model.save(os.path.join('./', 'my_model_ResNet.h5'))

model.save("ResNet50V22.h5", overwrite=True)
2
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