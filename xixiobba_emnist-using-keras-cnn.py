# Import modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import cv2

#keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras.utils import np_utils
import sklearn.metrics as metrics
#112799张训练集  18799张测试集  一共47个类  28*28+1=785  多出的1表示对应的label
train = pd.read_csv("../input/emnist-balanced-train.csv",delimiter = ',')
test = pd.read_csv("../input/emnist-balanced-test.csv", delimiter = ',')
mapp = pd.read_csv("../input/emnist-balanced-mapping.txt", delimiter = ' ', \
                   index_col=0, header=None, squeeze=True)
print("Train: %s, Test: %s, Map: %s" %(train.shape, test.shape, mapp.shape))
# Constants  每张图片的高和宽
HEIGHT = 28
WIDTH = 28
# Split x and y   取出图片和对应的label分别赋给train_x train_y
train_x = train.iloc[:,1:]
train_y = train.iloc[:,0]
del train

test_x = test.iloc[:,1:]
test_y = test.iloc[:,0]
del test
#28*28=784   一共112799张训练图片 所以 train_y存放了112799个对应的label 测试集同理
print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
#每张reshape到（28，28）同时做一个数据增强：水平翻转、旋转90度 作用是增加训练样本的多样性 ， 提高模型鲁棒性，避免过拟合。
def rotate(image):
    image = image.reshape([HEIGHT, WIDTH])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image
# Flip and rotate image
#对每个图片应用上面写的函数  
train_x = np.asarray(train_x)
train_x = np.apply_along_axis(rotate, 1, train_x)
print ("train_x:",train_x.shape)

test_x = np.asarray(test_x)
test_x = np.apply_along_axis(rotate, 1, test_x)
print ("test_x:",test_x.shape)
# Normalise
#归一化像素值到到0到1的范围内方便学习
train_x = train_x.astype('float32')
train_x /= 255
test_x = test_x.astype('float32')
test_x /= 255

# plot image
#显示九张图片 对应的title为label
for i in range(100, 109):
    plt.subplot(330 + (i+1))
    plt.imshow(train_x[i], cmap=plt.get_cmap('gray'))
    plt.title(chr(mapp[train_y[i]]))
# number of classes
num_classes = train_y.nunique()
# One hot encoding
#对label做一个one-hot编码 one hot编码就是将类别变量转换为机器学习算法易于利用的一种形式的过程。
train_y = np_utils.to_categorical(train_y, num_classes)
test_y = np_utils.to_categorical(test_y, num_classes)
print("train_y: ", train_y.shape)
print("test_y: ", test_y.shape)
# Reshape image for CNN  
# cnn输入形式为4D  first dimension=-1表示Batch_size是动态变化的  如果first dimension=1 每次喂给网络一张图片
train_x = train_x.reshape(-1, HEIGHT, WIDTH, 1)
test_x = test_x.reshape(-1, HEIGHT, WIDTH, 1)
# partition to train and val  从训练集中划分一部分充当验证集
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size= 0.10, random_state=7)
#模型 lenet-5
model = Sequential()

model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(AveragePooling2D())

model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D())

model.add(Flatten())

model.add(Dense(units=120, activation='relu'))

model.add(Dense(units=84, activation='relu'))

model.add(Dense(units=num_classes, activation = 'softmax'))
model.summary()
# Building model
# 搭建模型
# model = Sequential()

# model.add(Conv2D(filters=128, kernel_size=(5,5), padding = 'same', activation='relu',\
#                  input_shape=(HEIGHT, WIDTH,1)))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# model.add(Conv2D(filters=64, kernel_size=(3,3) , padding = 'same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Flatten())
# model.add(Dense(units=128, activation='relu'))
# model.add(Dropout(.5))
# model.add(Dense(units=num_classes, activation='softmax'))

# model.summary()
#编译模型以供训练 优化器为Adam 目标函数为交叉熵 评价指标为准确度
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#训练模型  fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况
history = model.fit(train_x, train_y, epochs=20, batch_size=512, verbose=1, \
                    validation_data=(val_x, val_y))
# plot accuracy and loss
def plotgraph(epochs, acc, val_acc):
    # Plot training & validation accuracy values
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
#%%
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
# Accuracy curve
plotgraph(epochs, acc, val_acc)
# loss curve
plotgraph(epochs, loss, val_loss)
#在测试集上模型的误差
score = model.evaluate(test_x, test_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
y_pred = model.predict(test_x)
y_pred = (y_pred > 0.5)
cm = metrics.confusion_matrix(test_y.argmax(axis=1), y_pred.argmax(axis=1))
cm