# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
import numpy as np
import tensorflow as tf

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image

from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D
from keras.layers import MaxPool2D,Flatten,Dropout,ZeroPadding2D,BatchNormalization
from keras.utils import np_utils
import keras
from keras.models import save_model,load_model
from keras.models import Model 
from keras.initializers import glorot_uniform

import pickle
import gzip
# 文件读取函数
def load_data():
    with gzip.open('../input/mnist-for-tf/mnist.pkl.gz') as fp:
        training_data, valid_data, test_data = pickle.load(fp, encoding='bytes')
    return training_data, valid_data, test_data

# 读取数据
training_data_0, valid_data_0, test_data_0 = load_data()

# 将向量转化为图片
train_data = [training_data_0[0].reshape([len(training_data_0[0]), 28, 28, 1])]
valid_data = [valid_data_0[0].reshape([len(valid_data_0[0]), 28, 28, 1])]
test_data = [test_data_0[0].reshape([len(test_data_0[0]), 28, 28, 1])]

# 将label转换为one-hot
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc.fit([[i] for i in range(10)])
train_data.append(enc.transform([[training_data_0[1][i]] for i in range(len(training_data_0[1]))]).toarray())
valid_data.append(enc.transform([[valid_data_0[1][i]] for i in range(len(valid_data_0[1]))]).toarray())
test_data.append(enc.transform([[test_data_0[1][i]] for i in range(len(test_data_0[1]))]).toarray())
# 建立CNN模型

def cnn_model(input_shape):
    # input placeholder
    X_input = Input(input_shape)    
    
    # layer0 :  conv->bn->relu->maxpool
    #X = Conv2D(32, (5,5), strides=(1, 1), padding = 'same', name = 'conv00', kernel_initializer = glorot_uniform(seed=0))(X_input)
    #X = Conv2D(32, (5,5), strides=(1, 1), padding = 'same', name = 'conv01', kernel_initializer = glorot_uniform(seed=0))(X_input)
    #X = BatchNormalization(axis = 3, name = 'bn0')(X)
    #X = Activation('relu')(X)
    #X = MaxPooling2D((3,3), name = 'maxpool0')(X)
    
    # layer1 :  conv->bn->relu->maxpool
    #X = Conv2D(64, (3,3), strides=(1, 1), padding = 'same', name = 'conv10', kernel_initializer = glorot_uniform(seed=0))(X)
    #X = Conv2D(64, (3,3), strides=(1, 1), padding = 'same', name = 'conv11', kernel_initializer = glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis = 3, name = 'bn1')(X)
    #X = Activation('relu')(X)
    #X = MaxPooling2D((3,3), name = 'maxpool1')(X)
    
    # FPGA Code Test
    X = Conv2D(8, (3,3), strides=(1, 1), padding = 'same', name = 'conv00', kernel_initializer = glorot_uniform(seed=0))(X_input)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3), name = 'maxpool1')(X)
    
    X = Conv2D(16, (3,3), strides=(1, 1), padding = 'same', name = 'conv01', kernel_initializer = glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3), name = 'maxpool2')(X)
    
    # FC-FC
    X = Flatten()(X)
    #X = Dense(96, activation = 'relu', name = 'fc0')(X)
    X = Dense(10, activation = 'softmax', name = 'fc1')(X)
    
    model = Model(inputs = X_input, outputs = X, name = 'MNISTModel')
    
    return model

#model = cnn_model((28, 28, 1))
#model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#model.fit(x = train_data[0], y = train_data[1], epochs = 20, batch_size = 128)

# 测试集成绩
#print(model.evaluate(x = test_data[0], y = test_data[1]))

# 保存模型
#model.save('my_model.h5')
#model.summary()
# 读取模型
model = load_model('../input/model5/my_model.h5')
model.summary()
# Kears直接输出神经网络权值
weights = np.array(model.get_weights())
conv_0, bias_0, conv_1, bias_1, fc, bias_2 = [weights[i] for i in range(6)]

os.makedirs('output/weights_save/conv0/') 
os.makedirs('output/weights_save/conv1/') 
os.makedirs('output/weights_save/fc/') 

# conv0 weights
for i in range(conv_0.shape[3]):
    np.savetxt("output/weights_save/conv0/W0"+str(i)+".csv", conv_0[:,:,:,i].reshape(3,3), delimiter=",")
np.savetxt("output/weights_save/conv0/bias_0.csv", bias_0, delimiter=",")

# conv1 weights
for k in range(conv_1.shape[3]):
    for c in range(conv_1.shape[2]):
       np.savetxt("output/weights_save/conv1/W0"+str(k)+str(c)+".csv", conv_1[:,:,c,k].reshape(3,3), delimiter=",") 
np.savetxt("output/weights_save/conv1/bias_11.csv", bias_1, delimiter=",")

# fc weights
np.savetxt("output/weights_save/fc/fc.csv", fc, delimiter=",")
np.savetxt("output/weights_save/fc/bias_21.csv", bias_2, delimiter=",")
# 读取MNIST竞赛测试集
test_data_kaggle = pd.read_csv('../input/digit-recognizer/test.csv')
test_data_kaggle = np.array(test_data_kaggle)
test_data_kaggle = np.reshape(test_data_kaggle, [len(test_data_kaggle), 28, 28, 1])
test_data_kaggle.shape

# 预测结果
pred_kaggle = model.predict(test_data_kaggle)
pred_kaggle = np.argmax(pred_kaggle, axis = 1)

result = list(range(1, len(pred_kaggle)+1))
result = {'ImageId': result, 'Label': pred_kaggle.tolist()}
result = pd.DataFrame(result)
result.to_csv('submit.csv', index = None)
from matplotlib import pyplot as plt

#绘制预测结果
test_data_plot = train_data[0][0:0+6]
test_data_pred = model.predict(test_data_plot)

test_data_pred = np.argmax(test_data_pred,axis=1)

Nrows = 2
Ncols = 3
for i in range(6):
    plt.subplot(Nrows, Ncols, i+1)
    plt.imshow(test_data_plot[i].reshape([28,28]), cmap='Greys_r')
    plt.title(' Pred: ' + str(test_data_pred[i]),
              fontsize=10)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
# 输出每层结果
from keras import backend as K

inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions

# Testing
test = train_data[0][0].reshape(1,28,28,1)
test[test>0] = 1
layer_outs = [func([test]) for func in functors]

