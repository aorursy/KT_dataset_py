# 该Python 3环境预装了许多实用的统计分析包

# kaggle/python docker镜像为: https://github.com/kaggle/docker-python

# 下面的代码导入了一些常用的把： 



# 用于线性代数等数学运算

import numpy as np 



# 数据梳理，CSV文件输入输出(如：pd.read_csv)

import pandas as pd



# 数据文件位于"/kaggle/input"目录，我们称之为“输入目录”

# 运行下面的代码会列出输入目录中的所有文件



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# 加载数据

def load_data(path):

    data = np.load(path)

    x_train = data['x_train']

    y_train = data['y_train']

    x_test = data['x_test']

    y_test = data['y_test']

    data.close()

    return (x_train, y_train), (x_test, y_test)
# 加载样本数据

(x_train, y_train), (x_test, y_test) = load_data('/kaggle/input/mnist-numpy/mnist.npz')

# 输出训练数据集统计信息

print("***************************************")

print("训练数据集")

print("***************************************")

print("训练样本数量 : " + str(x_train.shape[0]))

print("X_train维度 : " + str(x_train.shape))

print("Y_train维度 : " + str(y_train.shape))

print("***************************************")

print("测试数据集")

print("***************************************")

print("测试样本数量 : " + str(x_test.shape[0]))

print("X_test 维度 : " + str(x_test.shape))

print("Y_test 维度 : " + str(y_test.shape))
import matplotlib.pyplot as plt

# 绘制前5个样本图片

def draw_some_sample(x_train):

    plt.rcParams['font.sans-serif'] = ['SimHei']

    for i in range(5):

        fig = plt.figure(frameon=False)

        fig.set_size_inches(0.56, 0.56)



        ax = plt.Axes(fig, [0., 0., 1., 1.])

        ax.set_axis_off()

        fig.add_axes(ax)



        ax.imshow(x_train[i], cmap=plt.cm.gray, interpolation='nearest')

        fig.show()





# 输出一个样本像素值

def print_one_sample_pixels(sample):

    for i in range(28):

        line = ''

        for j in range(28):

            if sample[i][j] == 0:

                line = line + " "*(5-len(str(sample[i][j]))) + str(sample[i][j])

            else:

                line = line + " "*(5-len(str(sample[i][j]))) + str(sample[i][j])

        line = line + '\n'

        print(line)





# 绘制一个样本图片

def draw_one_sample(sample):

    fig = plt.figure(frameon=False)

    fig.set_size_inches(0.56, 0.56)



    ax = plt.Axes(fig, [0., 0., 1., 1.])

    ax.set_axis_off()

    fig.add_axes(ax)



    ax.imshow(sample, cmap=plt.cm.gray, interpolation='nearest')

    fig.show()
# 绘制前5个训练样本图像

draw_some_sample(x_train)



# 输出一个样本像素值

print_one_sample_pixels(x_train[1])
from keras.layers import Dense, Input

from keras.models import Model



# 关闭FutureWarning

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



# 定义神经网络模型

def my_model():

    # 输入层，28*28=784;

    inputs = Input(shape=(784,))



    # 隐藏层1，16个神经元

    x = Dense(16, activation='relu')(inputs)



    # 隐藏层2，16个神经元

    x = Dense(16, activation='relu')(x)



    # 输出层

    outputs = Dense(10, activation="softmax")(x)



    model = Model(inputs, outputs)



    return model
from keras.losses import categorical_crossentropy

from keras.optimizers import Adadelta

import numpy as np



# 训练模型

def train():

    # 读取数据集

    (x_train, y_train), (x_test, y_test) = load_data('/kaggle/input/mnist-numpy/mnist.npz')



    # 数据变维

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])



    y_train = (np.arange(10) == y_train[:, None]).astype(int)

    y_test = (np.arange(10) == y_test[:, None]).astype(int)



    # my_model实例化

    model = my_model()



    # 编译模型

    model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])



    # 训练模型

    model.fit(x_train, y_train, batch_size=6000, epochs=100, shuffle=True, verbose=1, validation_split=0.3, validation_data=(x_test, y_test))



    # 保存模型

    model.save('model.h5')



    # 用测试集评估模型

    predict = model.evaluate(x_test, y_test, batch_size=200)

    print()

    print("测试集损失值 = " + str(predict[0]))

    print("测试集正确率 = " + str(predict[1]))





# 运行训练

if __name__ == "__main__":

    train()
# 查看模型文件model.h5的保存位置”



import os

for dirname, _, filenames in os.walk('/kaggle'):

    for filename in filenames:

        print(os.path.join(dirname, filename))