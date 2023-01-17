# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# 样本文件路径

SAMPLE_FILE_PATH = "/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/icml_face_data.csv"



# 分类数量

NUM_CLASSES = 7



# 训练、校验、测试数据集HDF5文件的输出路径

TRAIN_HDF5 = "train.hdf5"

VAL_HDF5 = "val.hdf5"

TEST_HDF5 = "test.hdf5"



# 每批次样本数量

BATCH_SIZE = 128



# 项目输出文件保存目录

OUTPUT_PATH = "/kaggle/working"



# 模型保存位置及文件名称

MODEL_FILE = OUTPUT_PATH + "/model.h5"
import os

import h5py





# 定义HDF5DatasetWriter类

class HDF5DatasetWriter:

    """

    dims参数用来控制将要保存到数据集中的数据维度，类似于NumPy数组的shape。



    如果我们要保存扁平化的28 × 28 = 784 MNIST数据集原始像素数据，则dims=(70000, 784)，

    因为MNIST数据集共有70,000个样本，每个样本的维度是784。



    如果我们要存储原始CIFAR-10图像，则dims=(60000, 32, 32, 3)

    因为CIFAR-10数据集共有60,000图像，每个样本表示为一个32 × 32 × 3 RGB图像。

    """



    def __init__(self, dims, output_path, data_key="images", buf_size=1000):

        # 检查输出路径是否存在，如果存在则抛出异常

        if os.path.exists(output_path):

            raise ValueError('您提供的输出文件{}已经存在，请先手工删除！'.format(output_path))



        # 创建并打开可写入HDF5文件

        # 然后在其中创建两个数据集，一个用于存储图像/特征，另一个用于存储分类标签

        self.db = h5py.File(output_path, "w")

        self.data = self.db.create_dataset(data_key, dims, dtype="float")

        self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")



        # 保存缓存大小，然后初始化存缓和数据集索引

        self.buf_size = buf_size

        self.buffer = {"data": [], "labels": []}

        self.idx = 0



    def add(self, raw, label):

        # 将数据和标签添加到缓存

        self.buffer["data"].extend(raw)

        self.buffer["labels"].extend(label)



        if len(self.buffer["data"]) >= self.buf_size:

            self.flush()



    def flush(self):

        # 将缓存内容写入磁盘文件，然后清空缓存

        i = self.idx + len(self.buffer["data"])

        self.data[self.idx:i] = self.buffer["data"]

        self.labels[self.idx:i] = self.buffer["labels"]

        self.idx = i

        self.buffer = {"data": [], "labels": []}



    def store_class_labels(self, class_labels):

        # 创建一个数据集用来存储分类标签名称，然后保存分类标签

        dt = h5py.special_dtype(vlen=str)

        label_dim = (len(class_labels),)

        label_set = self.db.create_dataset("label_names", label_dim, dtype=dt)

        label_set[:] = class_labels



    def close(self):

        # 检查缓存中是否有记录，如果有，则必须写入磁盘文件

        if len(self.buffer["data"]) > 0:

            self.flush()



        # 关闭数据集

        self.db.close()

# 导入必要的包

import numpy as np



print("[信息] 加载CSV格式数据集文件...")

# 打开CSV格式数据集文件

file = open(SAMPLE_FILE_PATH)



# 跳过第一行（表头）

file.__next__()



# 声明训练、校验、测试数据集

(train_images, train_labels) = ([], [])

(val_images, val_labels) = ([], [])

(test_images, test_labels) = ([], [])



count_by_label_train = {}

count_by_label_val = {}

count_by_label_test = {}

# 遍历一遍文件的每一行

for row in file:

    # 提取每一行的标签（label）, 用途（usage）以及图像（image）。

    (label, usage, image) = row.strip().split(",")

    # 标签转整数

    label = int(label)

    if NUM_CLASSES == 6:

        if label == 1:

            label = 0

        if label > 0:

            label -= 1

    # 将一维像素列表变维成48x48灰度图像

    image = np.array(image.split(" "), dtype="uint8")

    image = image.reshape((48, 48))



    # 如果用途为Training，添加到训练样本集

    if usage == "Training":

        train_images.append(image)

        train_labels.append(label)

        count = count_by_label_train.get(label, 0)

        count_by_label_train[label] = count + 1

    # 如果用途为PublicTest，添加到校验样本集

    elif usage == "PublicTest":

        val_images.append(image)

        val_labels.append(label)

        count = count_by_label_val.get(label, 0)

        count_by_label_val[label] = count + 1

    # 如果用途为PrivateTest，添加到测试样本集

    elif usage == "PrivateTest":

        test_images.append(image)

        test_labels.append(label)

        count = count_by_label_test.get(label, 0)

        count_by_label_test[label] = count + 1



# 关闭CVS样本文件

file.close()



# 输出训练、校验、测试样本数量

print("[信息] 训练样本数量：{}".format(len(train_images)))

print("[信息] 校验样本数量：{}".format(len(val_images)))

print("[信息] 测试样本数量：{}".format(len(test_images)))

print("[信息] 训练样本分布：")

print(count_by_label_train)

print("[信息] 校验样本分布：")

print(count_by_label_val)

print("[信息] 测试样本分布：")

print(count_by_label_test)





# 构建一个训练、校验、测试数据集列表，

# 每个元素由数据集类型名称、全部样本文件名称、全部样本整型标签、HDF5输出文件构成

datasets = [(train_images, train_labels, TRAIN_HDF5),

            (val_images, val_labels, VAL_HDF5),

            (test_images, test_labels, TEST_HDF5)]



# 遍历数据集元组

for (images, labels, outputPath) in datasets:

    # 创建HDF5人写入器

    print("[信息] 构建 {}...".format(outputPath))

    writer = HDF5DatasetWriter((len(images), 48, 48), outputPath)



    # 遍历每个图像，将其加入数据集

    for (image, label) in zip(images, labels):

        writer.add([image], [label])



    # 关闭HDF5人写入器

    writer.close()

from tensorflow.keras.preprocessing.image import img_to_array





# 定义ImageToArrayPreprocessor类

class ImageToArrayPreprocessor:

    def __init__(self, data_format=None):

        # 保存图像数据的格式

        self.data_format = data_format



    def preprocess(self, image):

        """

        重置图像image的维度



        Args:

            image:  要预处理的图像



        Returns:

            维度重置后的图像

        """



        # 调用tensorflow.keras的img_to_array方法正确重置图像的维度

        return img_to_array(image, data_format=self.data_format)

from tensorflow.python.keras.utils import np_utils

import numpy as np

import h5py





class HDF5DatasetGenerator:

    def __init__(self,

                 db_file,

                 batch_size,

                 preprocessors=None,

                 aug=None,

                 binarize=True,

                 classes=2):

        # 每批次样本数量

        self.batchSize = batch_size



        # 预处理器列表

        self.preprocessors = preprocessors



        # 数据增强处理器列表，可以使用 Keras ImageDataGenerator实现的数据增强算法

        self.aug = aug



        # 标签是否二值化，我们在HDF5数据集中保存的类别标签为单个整型的列表，

        # 然而，当我们使用分类交叉商（categorical cross-entropy）

        # 或二值交叉商（binary cross-entropy）

        # 作为计算损失的方法，我们必须将标签二值化为独热编码向量组（one-hot encoded vectors）

        self.binarize = binarize



        # 不重复的类别数量，在计算标签二值化独热编码向量组时需要该值

        self.classes = classes



        # 打开HDF5数据集文件

        self.db = h5py.File(db_file, "r")



        # 数据集样本数量

        self.numImages = self.db["labels"].shape[0]



    def generator(self, passes=np.inf):

        # 初始化训练趟数计数器

        epochs = 0



        # 执行无限循环，一旦达到规定的训练趟数，模型会自动停止训练

        while epochs < passes:

            # 遍历HDF5数据集

            for i in np.arange(0, self.numImages, self.batchSize):

                # 从HDF5数据集取出一批样本和标签

                images = self.db["images"][i: i + self.batchSize]

                labels = self.db["labels"][i: i + self.batchSize]



                # 检查标签是否有转化为独热编码向量组

                if self.binarize:

                    labels = np_utils.to_categorical(labels, self.classes)



                # 如果有预处理器

                if self.preprocessors is not None:

                    # 初始化预处理结果图像列表

                    processed_images = []



                    # 遍历图像

                    for image in images:

                        # 遍历预处理器，对每个图像执行全部预处理

                        for p in self.preprocessors:

                            image = p.preprocess(image)



                        # 更新预处理结果图像列表

                        processed_images.append(image)



                    # 将图像数组更新为预处理结果图像

                    images = np.array(processed_images)



                # 如果指定了数据增强器，则实施之

                if self.aug is not None:

                    (images, labels) = next(self.aug.flow(images,

                                                          labels,

                                                          batch_size=self.batchSize))



                # 生成图像和标记元祖

                yield images, labels



            # 增加训练趟数计数器

            epochs += 1



    def close(self):

        # 关闭HDF5数据集

        self.db.close()
# 导入必须的包

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Activation

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Dense

from tensorflow.keras.regularizers import l2

from tensorflow.keras import backend





# 定义 MiniVGG16类

class MiniVGG16:

    @staticmethod

    def build(width, height, channel, classes, reg=0.0002):

        """

        根据输入样本的维度（width、height、channel），分类数量，正则化因子创建MiniVGG16网络模型

        Args:

            width:   输入样本的宽度

            height:  输入样本的高度

            channel: 输入样本的通道

            classes: 分类数量

            reg:     正则化因子



        Returns:

            AlexNet网络模型对象



        """



        model = Sequential(name="MiniVGG13")



        # 缺省输入格式为通道后置（"channels_last"）

        shape = (height, width, channel)

        channel_dimension = -1



        # 如果输入格式为通道前置

        # 重新设置输入格式和通道位置指示

        if backend.image_data_format() == "channels_first":

            shape = (channel, height, width)

            channel_dimension = 1



        # 第一卷积块

        model.add(Conv2D(64, (3, 3), input_shape=shape, padding="same", kernel_regularizer=l2(reg)))

        model.add(Activation("relu"))

        model.add(Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(reg)))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=channel_dimension))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Dropout(0.35))



        # 第二卷积块

        model.add(Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(reg)))

        model.add(Activation("relu"))

        model.add(Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(reg)))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=channel_dimension))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Dropout(0.35))



        # 第三卷积块

        model.add(Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2(reg)))

        model.add(Activation("relu"))

        model.add(Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2(reg)))

        model.add(Activation("relu"))

        model.add(Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2(reg)))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=channel_dimension))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Dropout(0.35))



        # 第四卷积块

        model.add(Conv2D(512, (3, 3), padding="same", kernel_regularizer=l2(reg)))

        model.add(Activation("relu"))

        model.add(Conv2D(512, (3, 3), padding="same", kernel_regularizer=l2(reg)))

        model.add(Activation("relu"))

        model.add(Conv2D(512, (3, 3), padding="same", kernel_regularizer=l2(reg)))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=channel_dimension))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Dropout(0.35))

        

        # 第一全连接层

        model.add(Flatten())

        model.add(Dense(64, kernel_regularizer=l2(reg)))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=channel_dimension))

        model.add(Dropout(0.35))



        # 第二全连接层

        model.add(Dense(64, kernel_regularizer=l2(reg)))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=channel_dimension))

        model.add(Dropout(0.35))



        # 第三全连接层

        model.add(Dense(classes, kernel_regularizer=l2(reg)))

        model.add(Activation("softmax"))



        return model





# 测试MiniVGG13类实例化并输出MiniVGG13模型的概要信息

if __name__ == "__main__":

    model = MiniVGG16.build(width=48, height=48, channel=1, classes=7, reg=0.0002)

    print(model.summary())
from tensorflow.keras.callbacks import ReduceLROnPlateau

learn_rate_fuc = ReduceLROnPlateau(monitor='val_accuracy',

                                   patience=3,

                                   verbose=1,

                                   factor=0.5,

                                   min_lr=0.000005)
# 导入必要的包

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam, Adamax

from tensorflow.keras.models import load_model

import tensorflow.keras.backend as K



import argparse

import json

import os



# 设置matplotlib可以在后台保存绘图

# matplotlib.use("Agg")





# 训练集数据增强生成器

train_aug = ImageDataGenerator(rotation_range=10,

                               zoom_range=0.1,

                               horizontal_flip=True,

                               rescale=1 / 255.0,

                               fill_mode="nearest")

# 校验集数据增强生成器

val_aug = ImageDataGenerator(rescale=1 / 255.0)



# 初始化图像预处理器

iap = ImageToArrayPreprocessor()



# 初始化训练数据集生成器

train_gen = HDF5DatasetGenerator(TRAIN_HDF5,

                                 BATCH_SIZE,

                                 aug=train_aug,

                                 preprocessors=[iap],

                                 classes=NUM_CLASSES)



# 初始化校验数据集生成器

val_gen = HDF5DatasetGenerator(VAL_HDF5,

                               BATCH_SIZE,

                               aug=val_aug,

                               preprocessors=[iap],

                               classes=NUM_CLASSES)



print("[信息] 编译模型...")

# 初始化优化器

# opt = Adam(lr=1e-3)

opt = Adamax()

model = MiniVGG16.build(width=48, height=48, channel=1, classes=NUM_CLASSES)

model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])



EPOCHS = 50



# 训练网络

history = model.fit_generator(train_gen.generator(),

                              steps_per_epoch=train_gen.numImages // BATCH_SIZE,

                              validation_data=val_gen.generator(),

                              validation_steps=val_gen.numImages // BATCH_SIZE,

                              epochs=EPOCHS,

                              max_queue_size=BATCH_SIZE * 2,

                              callbacks=[learn_rate_fuc],

                              verbose=1)



# 将训练得到的模型保存到文件

print("[信息] 保存模型...")

model.save(MODEL_FILE, overwrite=True)



# 关闭HDF5数据集

train_gen.close()

val_gen.close()

import matplotlib.pyplot as plt



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

t = f.suptitle('MiniVGG Training Performance', fontsize=12)

f.subplots_adjust(top=0.85, wspace=0.3)



epoch_list = list(range(1, EPOCHS + 1))

ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')

ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')

ax1.set_xticks(np.arange(0, EPOCHS + 1, 5))

ax1.set_ylabel('Accuracy Value')

ax1.set_xlabel('Epoch #')

ax1.set_title('Accuracy')

l1 = ax1.legend(loc="best")



ax2.plot(epoch_list, history.history['loss'], label='Train Loss')

ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')

ax2.set_xticks(np.arange(0, EPOCHS + 1, 5))

ax2.set_ylabel('Loss Value')

ax2.set_xlabel('Epoch #')

ax2.set_title('Loss')

l2 = ax2.legend(loc="best")