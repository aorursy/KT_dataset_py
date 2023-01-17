import keras
from keras.layers import Dense,Dropout, Activation, Flatten
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
# 读取数据
path = '../input/beijing-tavg/BeiJing annual TAVG.csv' #每一行的最后一个数据是目标值 一定要保证数据完整
dataset = pd.read_csv(path)
X = dataset["Year"]
Y = dataset["TAVG"]

model=Sequential() #建立顺序模型序列
model.add(Dense(units=1,input_dim=1))#输入维度为1，输出维度为1 添加一个网络层 输入维度为1，输出维度为1 
model.add(Activation('relu'))
model.add(Dense(units=1,input_dim=1))#输入维度为1，输出维度为1 添加一个网络层 输入维度为1，输出维度为1 
model.add(Activation('relu'))
model.compile(optimizer='adam',loss='mse') #设置SGD优化模型，

#训练，迭代步为3001次。

for step in range(3001):

    cost=model.train_on_batch(X,Y) #batch 为每次训练的批次

    if step%500 ==0:

        print('cost:',cost) #每500次输出一次

#打印权值和偏置值

w,b=model.layers[0].get_weights()

print("w:",w,"b:",b)
Y_pred = model.predict(X)
plt.scatter(X, Y)
plt.plot(X, Y_pred)
plt.show()