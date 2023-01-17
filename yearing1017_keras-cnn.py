import numpy as np

import pandas as pd

from keras.preprocessing.image import ImageDataGenerator, load_img # 数据增强

from keras.utils import to_categorical #

from sklearn.model_selection import train_test_split #

import matplotlib.pyplot as plt

import random

import os
# 超参数的设定

FAST_RUN = False

IMAGE_WIDTH=128

IMAGE_HEIGHT=128

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS=3
# 准备训练数据

filenames = os.listdir("/kaggle/input/dogs-vs-cats/train/train")

categories = [] # 存放类别

for filename in filenames:

    category = filename.split('.')[0] # 取出cat or dog

    if category == 'dog':

        categories.append(1)

    else:

        categories.append(0)

# 将文件与分好类别相对应        

df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})



df.head()
df['category'].value_counts().plot.bar() # 查看是否猫狗数量一致
from keras.models import Sequential

from keras.layers import *



model = Sequential()



model.add(Conv2D(32,(3,3),activation = 'relu',input_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS))) #卷积层

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# 全连接层

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax')) # 2 猫狗二分类问题



model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])



model.summary() # 输出模型各层的具体参数情况
from keras.callbacks import EarlyStopping, ReduceLROnPlateau



earlystop = EarlyStopping(patience=10) # 如发现loss相比上一个epoch训练没有下降，则经过patience个epoch后停止训练。



learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', # 被监控的量

                                            patience=2,        # 当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发

                                            verbose=1,         # 日志打印格式： 进度条

                                            factor=0.5,        # 每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少

                                            min_lr=0.00001)    # min_lr：学习率的下限

callbacks = [earlystop, learning_rate_reduction]
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})  # 将0，1替换为字符串，后面的image genaretor用得到



# train_test_split函数用于将矩阵随机划分为训练子集和验证子集，并返回划分好的训练集测试集样本和验证集测试集标签。

# test_size 为测试集样本数目与原始样本数目之比；

# random_state 随机数种子

train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)

# reset_index:重置索引列

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)
train_df['category'].value_counts().plot.bar() # 查看训练集10000cat10000dog
validate_df['category'].value_counts().plot.bar() # 查看验证集2500cat和dog
total_train = train_df.shape[0] #20000条训练数据

total_validate = validate_df.shape[0]

batch_size=15
# 对训练集

# 数据增强操作：ImageDataGenerator()是keras.preprocessing.image模块中的图片生成器，同时也可以在batch中对数据进行增强，扩充数据集大小，增强模型的泛化能力。

train_datagen = ImageDataGenerator(

    rotation_range=15, # 旋转范围

    rescale=1./255,    # rescale参数指定将图像张量的数字缩放

    shear_range=0.1,   # float, 透视变换的范围

    zoom_range=0.2,    # 缩放范围

    horizontal_flip=True,   # 水平反转

    width_shift_range=0.1,  # 水平平移范围

    height_shift_range=0.1  # 垂直平移范围

)



# 返回值：一个生成 (x, y) 元组的 DataFrameIterator， 其中：

#         x 是一个包含一批尺寸为 (batch_size, *target_size, channels) 的图像样本的 numpy 数组，

#         y 是对应的标签的 numpy 数组。

train_generator = train_datagen.flow_from_dataframe(

    train_df,                                  # Pandas dataframe

    "/kaggle/input/dogs-vs-cats/train/train",  # 字符串，目标目录的路径，其中包含在 dataframe 中映射的所有图像

    x_col='filename',                          # 字符串，dataframe 中包含目标图像文件夹的目录的列

    y_col='category',                          # 字符串或字符串列表，dataframe 中将作为目标数据的列

    target_size=IMAGE_SIZE,                    # 整数元组 (height, width)。 所有找到的图都会调整到这个维度。

    class_mode='categorical',                  # "categorical" 将是 2D one-hot 编码标签

    batch_size=batch_size                      # 批量数据的尺寸

)
# 对验证集

# 验证集不用数据增强，只需缩放

validation_datagen = ImageDataGenerator(rescale=1./255)

# 生成对应的图像与标签的（x，y）的数组

validation_generator = validation_datagen.flow_from_dataframe(

    validate_df, 

    "/kaggle/input/dogs-vs-cats/train/train", 

    x_col='filename',

    y_col='category',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
# 将数据fit model

# Keras中的fit()函数传入的x_train和y_train是被完整的加载进内存的,当然用起来很方便.

# 但是如果我们数据量很大，那么是不可能将所有数据载入内存的，必将导致内存泄漏，这时候我们可以用fit_generator函数来进行训练

epochs=3 if FAST_RUN else 20 # 训练轮次

history = model.fit_generator(

    train_generator,    # 训练数据生成器，20000个（x,y）元组

    epochs=epochs,      # 轮次

    validation_data=validation_generator,         # 验证集生成器

    validation_steps=total_validate//batch_size,  # 仅当 validation_data 是一个生成器时才可用。 在停止前 generator 生成的总步数（样本批数）

    steps_per_epoch=total_train//batch_size,      # 一个 epoch 完成并开始下一个 epoch 之前从 generator 产生的总步数（批次样本）

    callbacks=callbacks                           # 在训练时调用的一系列回调函数

)
model.save_weights("model.h5")
test_filenames = os.listdir("/kaggle/input/dogs-vs-cats/test1/test1")

test_df = pd.DataFrame({

    'filename': test_filenames

})

nb_samples = test_df.shape[0]



test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    "/kaggle/input/dogs-vs-cats/test1/test1", 

    x_col='filename',

    y_col=None,

    class_mode=None,

    target_size=IMAGE_SIZE,

    batch_size=batch_size,

    shuffle=False

)



predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))





test_df['category'] = np.argmax(predict, axis=-1)



label_map = dict((v,k) for k,v in train_generator.class_indices.items())

test_df['category'] = test_df['category'].replace(label_map)



test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })



test_df['category'].value_counts().plot.bar()
submission_df = test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)

submission_df.to_csv('submission.csv', index=False)