# 导入必要的包

import tensorflow as tf

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPool2D

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import concatenate



# 核初始化

kernel_init = tf.keras.initializers.glorot_uniform()



# 偏置初始化

bias_init = tf.keras.initializers.Constant(value=0.2)





# 生成潜深模块（Inception Module）的函数

def inception_module(x,

                     filters_1x1,

                     filters_3x3_reduce,

                     filters_3x3,

                     filters_5x5_reduce,

                     filters_5x5,

                     filters_pool_proj,

                     name=None):

    """

    生成GoogleNet的潜深模块

    Args:

        x: 上一层输入

        filters_1x1:          1×1卷积核数量

        filters_3x3_reduce:   消解降维3x3卷积的1×1卷积核数量

        filters_3x3:          3×3卷积核数量

        filters_5x5_reduce:   消解降维5x5卷积的1×1卷积核数量

        filters_5x5:          5×5卷积核数量

        filters_pool_proj:    消解降维最大池化的1×1卷积核数量

        name:                 潜深模块名称



    Returns

        生成的潜深模块堆叠合并特征图

    """



    # 1×1卷积

    conv_1x1 = Conv2D(filters_1x1,

                      (1, 1),

                      padding='same',

                      activation='relu')(x)

    conv_1x1 = BatchNormalization()(conv_1x1)



    # 消解降维3x3卷积的1×1卷积

    conv_3x3 = Conv2D(filters_3x3_reduce,

                      (1, 1),

                      padding='same',

                      activation='relu')(x)

    conv_3x3 = BatchNormalization()(conv_3x3)



    # 3x3卷积

    conv_3x3 = Conv2D(filters_3x3,

                      (3, 3),

                      padding='same',

                      activation='relu')(conv_3x3)

    conv_3x3 = BatchNormalization()(conv_3x3)



    # 消解降维5x5卷积的1×1卷积

    conv_5x5 = Conv2D(filters_5x5_reduce,

                      (1, 1),

                      padding='same',

                      activation='relu')(x)

    conv_5x5 = BatchNormalization()(conv_5x5)



    # 5x5卷积

    conv_5x5 = Conv2D(filters_5x5, (5, 5),

                      padding='same',

                      activation='relu')(conv_5x5)

    conv_5x5 = BatchNormalization()(conv_5x5)



    # 最大池化

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)



    # 消解降维最大池化的1×1卷积

    pool_proj = Conv2D(filters_pool_proj,

                       (1, 1),

                       padding='same',

                       activation='relu')(pool_proj)

    pool_proj = BatchNormalization()(pool_proj)



    # 堆叠合并

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)



    return output

# 导入必须的包

import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPool2D

from tensorflow.keras.layers import AveragePooling2D

from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Dense





# 定义 GoogleNet类

class GoogleNet:

    @staticmethod

    def build(width, height, channel, classes):

        """

        根据输入样本的维度（width、height、channel），分类数量创建GoogleNet网络模型

        Args:

            width:   输入样本的宽度

            height:  输入样本的高度

            channel: 输入样本的通道

            classes: 分类数量



        Returns:

           GoogleNet网络模型对象



        """



        input_layer = Input(shape=(width, height, channel))



        # 核初始化

        kernel_init = tf.keras.initializers.glorot_uniform()



        # 偏置初始化

        bias_init = tf.keras.initializers.Constant(value=0.2)



        # 卷积

        x = Conv2D(64,

                   (7, 7),

                   padding='same',

                   strides=(2, 2),

                   activation='relu',

                   name='conv_1_7x7/2')(input_layer)

        x = BatchNormalization()(x)



        # 最大池化

        x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)



        # 卷积

        x = Conv2D(64,

                   (1, 1),

                   padding='same',

                   strides=(1, 1),

                   activation='relu',

                   name='conv_2a_3x3/1')(x)

        x = BatchNormalization()(x)



        # 卷积

        x = Conv2D(192,

                   (3, 3),

                   padding='same',

                   strides=(1, 1),

                   activation='relu',

                   name='conv_2b_3x3/1')(x)

        x = BatchNormalization()(x)



        # 最大池化

        x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)



        # 潜深模块

        x = inception_module(x,

                             filters_1x1=64,

                             filters_3x3_reduce=96,

                             filters_3x3=128,

                             filters_5x5_reduce=16,

                             filters_5x5=32,

                             filters_pool_proj=32,

                             name='inception_3a')



        # 潜深模块

        x = inception_module(x,

                             filters_1x1=128,

                             filters_3x3_reduce=128,

                             filters_3x3=192,

                             filters_5x5_reduce=32,

                             filters_5x5=96,

                             filters_pool_proj=64,

                             name='inception_3b')



        # 最大池化

        x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)



        # 潜深模块

        x = inception_module(x,

                             filters_1x1=192,

                             filters_3x3_reduce=96,

                             filters_3x3=208,

                             filters_5x5_reduce=16,

                             filters_5x5=48,

                             filters_pool_proj=64,

                             name='inception_4a')



        '''

        # 辅助分类器

        x1 = AveragePooling2D((5, 5), strides=3)(x)

        x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)

        x1 = Flatten()(x1)

        x1 = Dense(1024, activation='relu')(x1)

        x1 = Dropout(0.4)(x1)

        x1 = Dense(classes, activation='softmax', name='auxilliary_output_1')(x1)

        '''



        # 潜深模块

        x = inception_module(x,

                             filters_1x1=160,

                             filters_3x3_reduce=112,

                             filters_3x3=224,

                             filters_5x5_reduce=24,

                             filters_5x5=64,

                             filters_pool_proj=64,

                             name='inception_4b')



        # 潜深模块

        x = inception_module(x,

                             filters_1x1=128,

                             filters_3x3_reduce=128,

                             filters_3x3=256,

                             filters_5x5_reduce=24,

                             filters_5x5=64,

                             filters_pool_proj=64,

                             name='inception_4c')



        # 潜深模块

        x = inception_module(x,

                             filters_1x1=112,

                             filters_3x3_reduce=144,

                             filters_3x3=288,

                             filters_5x5_reduce=32,

                             filters_5x5=64,

                             filters_pool_proj=64,

                             name='inception_4d')



        '''

        # 辅助分类器

        x2 = AveragePooling2D((5, 5), strides=3)(x)

        x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)

        x2 = Flatten()(x2)

        x2 = Dense(1024, activation='relu')(x2)

        x2 = Dropout(0.3)(x2)

        x2 = Dense(classes, activation='softmax', name='auxilliary_output_2')(x2)

        '''



        # 潜深模块

        x = inception_module(x,

                             filters_1x1=256,

                             filters_3x3_reduce=160,

                             filters_3x3=320,

                             filters_5x5_reduce=32,

                             filters_5x5=128,

                             filters_pool_proj=128,

                             name='inception_4e')



        # 最大池化

        x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)



        # 潜深模块

        x = inception_module(x,

                             filters_1x1=256,

                             filters_3x3_reduce=160,

                             filters_3x3=320,

                             filters_5x5_reduce=32,

                             filters_5x5=128,

                             filters_pool_proj=128,

                             name='inception_5a')



        # 潜深模块

        x = inception_module(x,

                             filters_1x1=384,

                             filters_3x3_reduce=192,

                             filters_3x3=384,

                             filters_5x5_reduce=48,

                             filters_5x5=128,

                             filters_pool_proj=128,

                             name='inception_5b')



        # 全局平均池化

        x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)



        # 随机失活

        x = Dropout(0.40)(x)



        # 全连接

        x = Dense(classes, activation='softmax', name='output')(x)



        # 创建GoogleNet模型

        # return Model(input_layer, [x, x1, x2], name='inception_v1')

        return Model(input_layer, x, name='inception_v1')





# 测试GoogleNet类实例化并输出GoogleNet模型的概要信息

if __name__ == "__main__":

    model = GoogleNet.build(width=224, height=224, channel=3, classes=196)

    print(model.summary())

from tensorflow.keras.optimizers import SGD, Adam, Adamax

from tensorflow.keras.callbacks import LearningRateScheduler

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import math



# 训练样本全路径文件名称

train_dirs = '/kaggle/input/stanford-car-dataset-by-classes-folder/car_data/car_data/train'

# 测试样本全路径文件名称

test_dirs ='/kaggle/input/stanford-car-dataset-by-classes-folder/car_data/car_data/test'



# 初始化优化器

epochs = 30

batch_size = 128

initial_lrate = 0.01





#  随训练趟数降低学习率

def decay(epoch, steps=100):

    initial_lrate = 0.01

    drop = 0.96

    epochs_drop = 8

    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

    return lrate





# 初始化学习调度器

lr_scheduler = LearningRateScheduler(decay, verbose=1)





# 构造用于数据增强的训练图像生成器

train_datagen = ImageDataGenerator(rotation_range=20,

                                   zoom_range=0.15,

                                   width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

                                   height_shift_range=0.2, # randomly shift images vertically (fraction of total height))

                                   shear_range=0.15,

                                   horizontal_flip=True,

                                   rescale=1./255,

                                   fill_mode="nearest")  



val_datagen = ImageDataGenerator(rescale=1./255)





trainGen = train_datagen.flow_from_directory(

        train_dirs,

        target_size=(224, 224),

        batch_size=batch_size,

        shuffle=True)



valGen = val_datagen.flow_from_directory(

        test_dirs,

        target_size=(224, 224),      

        batch_size=batch_size,

        shuffle=True)





opt = SGD(lr=initial_lrate, momentum=0.9, nesterov=False)

#opt = Adamax()

model = GoogleNet.build(width=224, height=224, channel=3, classes=196)



model.compile(loss='categorical_crossentropy',

              optimizer=opt,

              metrics=['accuracy'])



callbacks = [lr_scheduler]

history = model.fit_generator(trainGen,

                              steps_per_epoch=8144 // batch_size,

                              epochs=epochs,

                              validation_data=valGen,

                              validation_steps=8041 // batch_size,

                              max_queue_size=batch_size * 2,

                              callbacks=callbacks,

                              verbose=1)
import matplotlib.pyplot as plt



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

t = f.suptitle('Basic CNN Performance', fontsize=12)

f.subplots_adjust(top=0.85, wspace=0.3)



epoch_list = list(range(1,31))

ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')

ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')

ax1.set_xticks(np.arange(0, 31, 5))

ax1.set_ylabel('Accuracy Value')

ax1.set_xlabel('Epoch #')

ax1.set_title('Accuracy')

l1 = ax1.legend(loc="best")



ax2.plot(epoch_list, history.history['loss'], label='Train Loss')

ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')

ax2.set_xticks(np.arange(0, 31, 5))

ax2.set_ylabel('Loss Value')

ax2.set_xlabel('Epoch #')

ax2.set_title('Loss')

l2 = ax2.legend(loc="best")