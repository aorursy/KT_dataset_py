import sys, os

sys.path.append('../input/deeplearningfromscratch/deeplearningfromscratch')  # 为了导入父目录中的文件而进行的设定



import numpy as np

x= np.array([1.0, 2.0, 3.0])

print(x)
type(x)
# 下面是NumPy数组的算术运算例子

x = np.array([1.0, 2.0, 3.0])

y = np.array([2.0, 3.0, 4.0])

print(x + y)

print(x - y)

print(x * y) # element-wise porduct

print(x / y)
x = np.array([1.0, 2.0, 3.0])

print(x / 2.0)
# NumPy的多维数组

A = np.array([[1, 2],[3, 4]])

print(A)
A.shape
A.dtype
# 矩阵的算术运算

B = np.array([[3, 0],[0,6]])

A+B
A*B
print(A)

A*10
# 一个广播的例子

A = np.array([[1, 2], [3, 4]])

B = np.array([10, 20])

A * B
# 访问元素

X = np.array([[51, 55],[14,19],[0,4]])

print(X)
for row in X:

    print(row)
# 使用数组访问各个元素

X = X.flatten() # 将X转换为一维数组

print(X)
X[np.array([0,2,4])] #获取索引为0、2、4的元素
# 从X中抽取大于15的元素

X > 15
X[X>15]
import numpy as np

import matplotlib.pyplot as plt
#生成数据

x = np.arange(0, 6, 0.1) #生成0-6的数据，步长为0.1

y = np.sin(x)

#绘制图像

plt.plot(x,y)

plt.show()
# 尝试追加cos函数的图形，并尝试使用pyplot的添加标题和x轴标签名等其他功能

y2 = np.cos(x)



plt.plot(x, y, label="sin")

plt.plot(x, y2, linestyle = "--", label="cos")#用虚线绘制

plt.xlabel("x")

plt.ylabel("Y")

plt.title('sin & cos')

plt.legend()

plt.show()
from matplotlib.image import imread
img = imread('../input/lixiangzhilu/1.jpg')
plt.imshow(img)
# 与门

def AND(x1, x2):

    w1, w2, theta = 0.5, 0.5, 0.7

    temp = x1*w1 + x2*w2

    if temp <= theta:

        return 0

    elif temp > theta:

        return 1
AND(0, 0)
AND(1,0)
AND(0,1)
AND(1,1)
# NumPy实现感知机与权重与偏置

import numpy as np

x = np.array([0, 1])#输入

w = np.array([0.5, 0.5])#权重

b = -0.7#偏置

w*x
np.sum(w*x)
np.sum(w*x)+b#大约为-0.2，由于浮点数小数造成运算误差
# 使用权重和偏置实现与门

def AND2(x1, x2):

    x = np.array([x1, x2])#输入

    w = np.array([0.5, 0.5])#权重

    b = -0.7#偏置

    tmp = np.sum(w*x) + b

    if tmp <= 0:

        return 0

    else:

        return 1
# 实现与非门

def NAND2(x1, x2):

    x = np.array([x1, x2])#输入

    w = np.array([-0.5, -0.5])#权重

    b = 0.7#偏置

    tmp = np.sum(w*x) + b

    if tmp <= 0:

        return 0

    else:

        return 1

# 实现或门

def OR2(x1, x2):

    x = np.array([x1, x2])#输入

    w = np.array([0.5, 0.5])#权重

    b = -0.2#偏置

    tmp = np.sum(w*x) + b

    if tmp <= 0:

        return 0

    else:

        return 1
# 异或门的实现

def XOR2(x1, x2):

    s1 = NAND2(x1, x2)

    s2 = OR2(x1, x2)

    y = AND2(s1, s2)

    return y
XOR2(0, 0)
XOR2(1, 0)
XOR2(0, 1)
XOR2(1,1)
import numpy as np

import matplotlib.pylab as plt

#实现简单的阶跃函数

def step_function0(x):

    if x > 0:

        return 1

    else:

        return 0



def step_function1(x):

    y = x >0

    return y.astype(np.int)



def step_function(x):

    return np.array(x > 0, dtype=np.int)



x = np.arange(-5.0, 5.0, 0.1)

y = step_function(x)

plt.plot(x, y)

plt.ylim(-0.1, 1.1) # 指定y轴的范围

plt.show()
#sigmoid函数

def sigmoid(x):

    return 1 / (1 + np.exp(-x))



yy = sigmoid(x)

plt.plot(x,yy)

plt.ylim(-0.1,1.1)

plt.show()
plt.plot(x,y,linestyle = "--",label="step")

plt.plot(x,yy,label="sigmoid")

plt.show()
#ReLU函数实现

def relu(x):

    return np.maximum(0,x)



y3 = relu(x)

plt.plot(x,y3)

plt.show()
# 一维数组

import numpy as np

A = np.array([1, 2, 3, 4])

print(A)
np.ndim(A)#获取数组维数
A.shape#获取数组的形状
A.shape[0]#其结果是个元组（tuple）
# 二维数组,也是矩阵

B = np.array([[1,2], [3,4], [5,6]])

print(B)
np.ndim(B)
B.shape
# 矩阵乘法

C = np.array([[1,2],[3,4]])

C.shape
D = np.array([[5,6],[7,8]])

D.shape
np.dot(C,D)#乘积为点积
np.dot(D,C)#绝大部分矩阵不满足乘法交换律
# 使用NumPy矩阵实现神经网络

X=np.array([1,2])

X.shape
W=np.array([[1,3,5],[2,4,6]])

print(W)
W.shape
YY=np.dot(X,W)

print(YY)
# 多维数组实现A(1)=XW(1)+B

# 输入层到第1层的信号传递

X=np.array([1.0, 0.5])

W1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])

B1=np.array([0.1,0.2,0.3])



print(W1.shape)

print(X.shape)

print(B1.shape)
A1=np.dot(X,W1)+B1

#观察第1层中激活函数的计算过程,激活函数使用sigmoid

Z1=sigmoid(A1)

print(A1)

print(Z1)
#第1层到第2层的信号传递

W2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])

B2=np.array([0.1,0.2])



print(Z1.shape)

print(W2.shape)

print(B2.shape)

A2=np.dot(Z1,W2) + B2

Z2=sigmoid(A2)
#第2层到输出层的信号传递

#定义输出层的激活函数为恒等函数

def identity_function(x):

    return x



W3=np.array([[0.1,0.3],[0.2,0.4]])

B3=np.array([0.1,0.2])

A3=np.dot(Z2,W3)+B3

Y=identity_function(A3)

print(Y)
# 3层网络实现案例，把权重记为大写字母W1，其他的偏置或中间结果等用小写字母表示

def init_network():

    network = {}

    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])

    network['b1'] = np.array([0.1,0.2,0.3])

    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])

    network['b2'] = np.array([0.1, 0.2])

    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])

    network['b3'] = np.array([0.1,0.2])

    

    return network



def forward(network, x):

    W1, W2, W3 = network['W1'], network['W2'], network['W3']

    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    

    a1 = np.dot(x, W1) + b1

    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2

    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3

    y = identity_function(a3)

    

    return y



network = init_network()

x = np.array([1.0, 0.5])

y = forward(network, x)

print(y)
#实现softmax函数

a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a) #指数函数

print(exp_a)

sum_exp_a = np.sum(exp_a) #指数函数的和

print(sum_exp_a)

y = exp_a / sum_exp_a

print(y)
# 定义softmax函数。供以后使用(初始版)

def softmax1(a):

    exp_a = np.exp(a)

    sum_exp_a = np.sum(exp_a)

    y = exp_a / sum_exp_a

    

    return y
# 演示计算溢出情况与解决

a = np.array([1010, 1000, 990])

np.exp(a) / np.sum(np.exp(a)) #softmax,并未正确被计算，报错
c = np.max(a) #1010

a - c

np.exp(a - c) / np.sum(np.exp(a - c))
# 定义softmax函数。供以后使用(正式用版)

def softmax(a):

    c = np.max(a)

    exp_a = np.exp(a - c)#溢出对策

    sum_exp_a = np.sum(exp_a)

    y = exp_a / sum_exp_a

    

    return y
# 可解释为“概率”的softmax函数

a = np.array([0.3, 2.9, 4.0])

y = softmax(a)

print(y)

np.sum(y)
# 代码暂略

import sys, os

print(sys.path.append(os.pardir))
# 均方误差定义函数

def mean_squared_error(y, t):

    return 0.5 * np.sum((y-t)**2)
# 设“2”为正解

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

#例1：“2”的概率最高的情况（0.6）

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

mean_squared_error(np.array(y),np.array(t))
# 例2：“7”的概率最高的情况（0.6）

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

mean_squared_error(np.array(y),np.array(t))
# 实现交叉熵误差

def cross_entropy_error0(y, t):

    delta = 1e-7

    return -np.sum(t * np.log(y + delta))
# 进行简单计算

# 设“2”为正解

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

#例1：“2”的概率最高的情况（0.6）

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

cross_entropy_error0(np.array(y),np.array(t))
# 例2：“7”的概率最高的情况（0.6）

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

cross_entropy_error0(np.array(y),np.array(t))
# 读入MNIST 代码暂略

import sys, os

sys.path.append(os.pardir)

import numpy as np

# from dataset.mnist import load_mnist
# 使用np.random.choice进行随机选取

np.random.choice(60000, 10)
# 可同时处理单个和批量数据

def cross_entropy_error1(y, t):

    if y.nidm == 1:

        t = t.reshape(1, t.size)

        y = y.reshape(1, y.size)



    delta = 1e-7

    batch_size = y.shape[0]

    return -np.sum(t * np.log(y + delta)) / batch_size
# 可同时处理单个和批量数据

def cross_entropy_error2(y, t):

    if y.nidm == 1:

        t = t.reshape(1, t.size)

        y = y.reshape(1, y.size)



    delta = 1e-7

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
np.float32(1e-50)
# 对舍入误差减小与使用中心差分实现函数导数程序

def numerical_diff(f, x):

    h = 1e-4 #0.0001

    return (f(x+h) - f(x-h)) / (2*h)
# 实现例子

def function_1(x):

    return 0.01*x**2 + 0.1*x



import numpy as np

import matplotlib.pylab as plt



x = np.arange(0.0, 20.0, 0.1) #以0.1为单位，从0到20的数组x

y = function_1(x)

plt.xlabel("x")

plt.ylabel("y")

plt.plot(x, y)

plt.show()
#计算上面式子的5,10处导数

numerical_diff(function_1, 5)
numerical_diff(function_1, 10)
def tangent_line(f, x):

    d = numerical_diff(f, x)

    print(d)

    y = f(x) - d*x

    return lambda t: d*t + y
x = np.arange(0.0, 20.0, 0.1)

y = function_1(x)

plt.xlabel("x")

plt.ylabel("f(x)")



tf = tangent_line(function_1, 5)

tf2 = tangent_line(function_1, 10)

y2 = tf(x)

y22 = tf2(x)



plt.plot(x, y)

plt.plot(x, y2)

plt.show()

plt.plot(x, y)

plt.plot(x, y22)

plt.show()
# 实现上式的代码

def function_2(x):

    #或者return np.sum(x**2)

    return x[0]**2 + x[1]**2
# 求偏导1

def function_tmp1(x0):

    return x0*x0 + 4.0**2.0



numerical_diff(function_tmp1, 3.0)
# 求偏导2

def function_tmp2(x1):

    return 3.0**2.0 + x1*x1



numerical_diff(function_tmp1, 4.0)
# 梯度的代码实现

def numerical_gradient(f, x):

    h = 1e-4 #0.001

    grad = np.zeros_like(x) #生成和x形状相同的数组

    

    for idx in range(x.size):

        tmp_val = x[idx]

        # f(x+h)的计算

        x[idx] = tmp_val + h

        fxh1 = f(x)

        

        # f(x-h)的计算

        x[idx] = tmp_val - h

        fxh2 = f(x)

        

        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # 还原值

        

    return grad
# 求点(3,4) (0,2) (3,0)处的梯度

numerical_gradient(function_2, np.array([3.0, 4.0]))
numerical_gradient(function_2, np.array([0.0, 2.0]))
numerical_gradient(function_2, np.array([3.0, 0.0]))
import numpy as np

import matplotlib.pylab as plt

from mpl_toolkits.mplot3d import Axes3D





def _numerical_gradient_no_batch(f, x):

    h = 1e-4 # 0.0001

    grad = np.zeros_like(x)

    

    for idx in range(x.size):

        tmp_val = x[idx]

        x[idx] = float(tmp_val) + h

        fxh1 = f(x) # f(x+h)

        

        x[idx] = tmp_val - h 

        fxh2 = f(x) # f(x-h)

        grad[idx] = (fxh1 - fxh2) / (2*h)

        

        x[idx] = tmp_val # 还原值

        

    return grad





def numerical_gradient(f, X):

    if X.ndim == 1:

        return _numerical_gradient_no_batch(f, X)

    else:

        grad = np.zeros_like(X)

        

        for idx, x in enumerate(X):

            grad[idx] = _numerical_gradient_no_batch(f, x)

        

        return grad





def function_2(x):

    if x.ndim == 1:

        return np.sum(x**2)

    else:

        return np.sum(x**2, axis=1)





def tangent_line(f, x):

    d = numerical_gradient(f, x)

    print(d)

    y = f(x) - d*x

    return lambda t: d*t + y

     

if __name__ == '__main__':

    x0 = np.arange(-2, 2.5, 0.25)

    x1 = np.arange(-2, 2.5, 0.25)

    X, Y = np.meshgrid(x0, x1)

    

    X = X.flatten()

    Y = Y.flatten()

    

    grad = numerical_gradient(function_2, np.array([X, Y]) )

    

    plt.figure()

    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")#,headwidth=10,scale=40,color="#444444")

    plt.xlim([-2, 2])

    plt.ylim([-2, 2])

    plt.xlabel('x0')

    plt.ylabel('x1')

    plt.grid()

    plt.legend()

    plt.draw()

    plt.show()
def gradient_descent(f, init_x, lr=0.01, step_num=100):

    """

    参数f是要进行最优化的函数，init_x是初始值，lr是学习率learning rate，step_num是梯度法的重复次数。

    numerical_gradient(f,x)会求函数的梯度，用该梯度乘以学习率得到的值进行更新操作，由step_num指定重复的次数。

    """

    x = init_x

    x_history = []



    for i in range(step_num):

        x_history.append( x.copy() )



        grad = numerical_gradient(f, x)

        x -= lr * grad



    return x, np.array(x_history)
# 用梯度法求f(x0+x1)=x0^2+x1^2的最小值

def function_2(x):

    return x[0]**2 + x[1]**2



init_x = np.array([-3.0, 4.0])

x, x_history = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
# 用图像表示上面的函数梯度下降法的步骤

plt.plot( [-5, 5], [0,0], '--b')

plt.plot( [0,0], [-5, 5], '--b')

plt.plot(x_history[:,0], x_history[:,1], 'o')



plt.xlim(-3.5, 3.5)

plt.ylim(-4.5, 4.5)

plt.xlabel("X0")

plt.ylabel("X1")

plt.show()
# 学习率过大的例子：lr=10.0

init_x = np.array([-3.0, 4.0])

x, x_history = gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100)

x
# 学习率过小的例子：lr=1e-10

init_x = np.array([-3.0, 4.0])

x, x_history = gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100)

x
# 一个简单的神经网络

# coding: utf-8

import sys, os

sys.path.append('../input/deeplearningfromscratch/deeplearningfromscratch')  # 为了导入父目录中的文件而进行的设定

import numpy as np

from common.functions import softmax, cross_entropy_error

from common.gradient import numerical_gradient





class simpleNet:

    def __init__(self):

        self.W = np.random.randn(2,3)



    def predict(self, x):

        return np.dot(x, self.W)



    def loss(self, x, t):

        z = self.predict(x)

        y = softmax(z)

        loss = cross_entropy_error(y, t)



        return loss
net = simpleNet()

print(net.W)



x = np.array([0.6, 0.9])

p = net.predict(x)

print(p)



print(np.argmax(p))



t = np.array([0, 0, 1]) # 正确解标签



f = lambda w: net.loss(x, t)

dW = numerical_gradient(f, net.W)



print(dW)
from common.functions import *

from common.gradient import numerical_gradient





class TwoLayerNet:



    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

        # 初始化权重

        self.params = {}

        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)

        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)

        self.params['b2'] = np.zeros(output_size)



    def predict(self, x):

        W1, W2 = self.params['W1'], self.params['W2']

        b1, b2 = self.params['b1'], self.params['b2']

    

        a1 = np.dot(x, W1) + b1

        z1 = sigmoid(a1)

        a2 = np.dot(z1, W2) + b2

        y = softmax(a2)

        

        return y

        

    # x:输入数据, t:监督数据

    def loss(self, x, t):

        y = self.predict(x)

        

        return cross_entropy_error(y, t)

    

    def accuracy(self, x, t):

        y = self.predict(x)

        y = np.argmax(y, axis=1)

        t = np.argmax(t, axis=1)

        

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

        

    # x:输入数据, t:监督数据

    def numerical_gradient(self, x, t):

        loss_W = lambda W: self.loss(x, t)

        

        grads = {}

        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])

        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])

        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])

        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        

        return grads

        

    def gradient(self, x, t):

        W1, W2 = self.params['W1'], self.params['W2']

        b1, b2 = self.params['b1'], self.params['b2']

        grads = {}

        

        batch_num = x.shape[0]

        

        # forward

        a1 = np.dot(x, W1) + b1

        z1 = sigmoid(a1)

        a2 = np.dot(z1, W2) + b2

        y = softmax(a2)

        

        # backward

        dy = (y - t) / batch_num

        grads['W2'] = np.dot(z1.T, dy)

        grads['b2'] = np.sum(dy, axis=0)

        

        da1 = np.dot(dy, W2.T)

        dz1 = sigmoid_grad(a1) * da1

        grads['W1'] = np.dot(x.T, dz1)

        grads['b1'] = np.sum(dz1, axis=0)



        return grads
# 二层神经网络例子

net = TwoLayerNet(input_size = 784, hidden_size = 100, output_size = 10)

print(net.params['W1'].shape)

print(net.params['b1'].shape)

print(net.params['W2'].shape)

print(net.params['b2'].shape)
#推理处理的实现如下

x = np.random.rand(100, 784) # 伪输入数据100笔

y = net.predict(x)

t = np.random.rand(100, 10) # 伪正确解标签10笔



# grads = net.numerical_gradient(x, t) # 计算梯度，使用传统的基于数值微分计算参数的梯度

grads = net.gradient(x, t) # 计算梯度，使用误差反向传播算法



print(grads['W1'].shape)

print(grads['b1'].shape)

print(grads['W2'].shape)

print(grads['b2'].shape)
import numpy as np

import matplotlib.pyplot as plt

from dataset.mnist import load_mnist

# from two_layer_net import TwoLayerNet



# 读入数据

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)



network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)



# 超参数

iters_num = 10000  # 适当设定循环的次数

train_size = x_train.shape[0]

batch_size = 100

learning_rate = 0.1



train_loss_list = []

train_acc_list = []

test_acc_list = []



# 平均每个epoch的重复次数

iter_per_epoch = max(train_size / batch_size, 1)



for i in range(iters_num):

    # 获取mini-batch

    batch_mask = np.random.choice(train_size, batch_size)

    x_batch = x_train[batch_mask]

    t_batch = t_train[batch_mask]

    

    # 计算梯度

    #grad = network.numerical_gradient(x_batch, t_batch)

    grad = network.gradient(x_batch, t_batch)

    

    # 更新参数

    for key in ('W1', 'b1', 'W2', 'b2'):

        network.params[key] -= learning_rate * grad[key]

    

    # 记录学习过程

    loss = network.loss(x_batch, t_batch)

    train_loss_list.append(loss)

    

    # 计算每个epoch的识别精度

    if i % iter_per_epoch == 0:

        train_acc = network.accuracy(x_train, t_train)

        test_acc = network.accuracy(x_test, t_test)

        train_acc_list.append(train_acc)

        test_acc_list.append(test_acc)

        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))



# 绘制图形

markers = {'train': 'o', 'test': 's'}

x = np.arange(len(train_acc_list))

plt.plot(x, train_acc_list, label='train acc')

plt.plot(x, test_acc_list, label='test acc', linestyle='--')

plt.xlabel("epochs")

plt.ylabel("accuracy")

plt.ylim(0, 1.0)

plt.legend(loc='lower right')

plt.show()
# 实现乘法层

class MulLayer:

    def __init__(self):

        self.x = None

        self.y = None



    def forward(self, x, y):

        self.x = x

        self.y = y                

        out = x * y



        return out



    def backward(self, dout):

        dx = dout * self.y

        dy = dout * self.x



        return dx, dy
apple = 100

apple_num = 2

tax = 1.1



mul_apple_layer = MulLayer()

mul_tax_layer = MulLayer()



# forward

apple_price = mul_apple_layer.forward(apple, apple_num)

price = mul_tax_layer.forward(apple_price, tax)



# backward

dprice = 1

dapple_price, dtax = mul_tax_layer.backward(dprice)

dapple, dapple_num = mul_apple_layer.backward(dapple_price)



print("price:", int(price))

print("dApple:", dapple)

print("dApple_num:", int(dapple_num))

print("dTax:", dtax)
# 实现加法层

class AddLayer:

    def __init__(self):

        pass #表示什么也不运行



    def forward(self, x, y):

        out = x + y



        return out



    def backward(self, dout):

        dx = dout * 1

        dy = dout * 1



        return dx, dy
apple = 100

apple_num = 2

orange = 150

orange_num = 3

tax = 1.1



# layer

mul_apple_layer = MulLayer()

mul_orange_layer = MulLayer()

add_apple_orange_layer = AddLayer()

mul_tax_layer = MulLayer()



# forward

apple_price = mul_apple_layer.forward(apple, apple_num)  # (1)

orange_price = mul_orange_layer.forward(orange, orange_num)  # (2)

all_price = add_apple_orange_layer.forward(apple_price, orange_price)  # (3)

price = mul_tax_layer.forward(all_price, tax)  # (4)



# backward

dprice = 1

dall_price, dtax = mul_tax_layer.backward(dprice)  # (4)

dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)  # (3)

dorange, dorange_num = mul_orange_layer.backward(dorange_price)  # (2)

dapple, dapple_num = mul_apple_layer.backward(dapple_price)  # (1)



print("price:", int(price))

print("dApple:", dapple)

print("dApple_num:", int(dapple_num))

print("dOrange:", dorange)

print("dOrange_num:", int(dorange_num))

print("dTax:", dtax)
# ReLU

class Relu:

    def __init__(self):

        self.mask = None



    def forward(self, x):

        self.mask = (x <= 0)

        out = x.copy()

        out[self.mask] = 0



        return out



    def backward(self, dout):

        dout[self.mask] = 0

        dx = dout



        return dx
x = np.array([[1.0, -0.5], [-2.0, 3.0]])

print(x)

mask = (x <= 0)

print(mask)
# Sigmoid

class Sigmoid:

    def __init__(self):

        self.out = None



    def forward(self, x):

        out = sigmoid(x)

        self.out = out

        return out



    def backward(self, dout):

        dx = dout * (1.0 - self.out) * self.out



        return dx
# 回顾

X = np.random.rand(2)

W = np.random.rand(2,3)

B = np.random.rand(3)



print(X)

print(W)

print(B)



print(X.shape)

print(W.shape)

print(B.shape)



Y = np.dot(X, W) + B

print(Y)
X_dot_W = np.array([[0, 0, 0], [10, 10, 10]])

B = np.array([1, 2, 3])

print(X_dot_W)

print(X_dot_W + B)
dY = np.array([[1, 2, 3],[4, 5, 6]])

print(dY)

dB = np.sum(dY, axis = 0)

print(dB)
# Affine的实现（考虑输入数据为张量（四维数据）的情况）

class Affine:

    def __init__(self, W, b):

        self.W =W

        self.b = b

        

        self.x = None

        self.original_x_shape = None

        # 权重和偏置参数的导数

        self.dW = None

        self.db = None



    def forward(self, x):

        # 对应张量

        self.original_x_shape = x.shape

        x = x.reshape(x.shape[0], -1)

        self.x = x



        out = np.dot(self.x, self.W) + self.b



        return out



    def backward(self, dout):

        dx = np.dot(dout, self.W.T)

        self.dW = np.dot(self.x.T, dout)

        self.db = np.sum(dout, axis=0)

        

        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）

        return dx
# Softmax-with-Loss层实现

class SoftmaxWithLoss:

    def __init__(self):

        self.loss = None

        self.y = None # softmax的输出

        self.t = None # 监督数据



    def forward(self, x, t):

        self.t = t

        self.y = softmax(x)

        self.loss = cross_entropy_error(self.y, self.t)

        

        return self.loss



    def backward(self, dout=1):

        batch_size = self.t.shape[0]

        if self.t.size == self.y.size: # 监督数据是one-hot-vector的情况

            dx = (self.y - self.t) / batch_size

        else:

            dx = self.y.copy()

            dx[np.arange(batch_size), self.t] -= 1

            dx = dx / batch_size

        

        return dx
# 误差反向传播法TwoLayerNet代码实现

import numpy as np

from common.layers import *

from common.gradient import numerical_gradient

from collections import OrderedDict





class TwoLayerNet:



    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):

        # 初始化权重

        self.params = {}

        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)

        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 

        self.params['b2'] = np.zeros(output_size)



        # 生成层 注意这里

        self.layers = OrderedDict() # 注意这里

        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])

        self.layers['Relu1'] = Relu()

        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])



        self.lastLayer = SoftmaxWithLoss()

        

    def predict(self, x):

        for layer in self.layers.values():

            x = layer.forward(x)

        

        return x

        

    # x:输入数据, t:监督数据

    def loss(self, x, t):

        y = self.predict(x)

        return self.lastLayer.forward(y, t)

    

    def accuracy(self, x, t):

        y = self.predict(x)

        y = np.argmax(y, axis=1)

        if t.ndim != 1 : t = np.argmax(t, axis=1)

        

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

        

    # x:输入数据, t:监督数据

    def numerical_gradient(self, x, t):

        loss_W = lambda W: self.loss(x, t)

        

        grads = {}

        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])

        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])

        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])

        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        

        return grads

        

    def gradient(self, x, t):

        # forward 注意这里

        self.loss(x, t)



        # backward 注意这里

        dout = 1

        dout = self.lastLayer.backward(dout)

        

        layers = list(self.layers.values())

        layers.reverse()

        for layer in layers:

            dout = layer.backward(dout)



        # 设定

        grads = {}

        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db

        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db



        return grads
# 误差反响传播法的梯度确认实现

import numpy as np

from dataset.mnist import load_mnist



# 读入数据

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)



network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)



x_batch = x_train[:3]

t_batch = t_train[:3]



grad_numerical = network.numerical_gradient(x_batch, t_batch)

grad_backprop = network.gradient(x_batch, t_batch)



for key in grad_numerical.keys():

    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )

    print(key + ":" + str(diff))
# 使用误差反向传播法的学习的实现

import numpy as np

from dataset.mnist import load_mnist



# 读入数据

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)



network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)



iters_num = 10000

train_size = x_train.shape[0]

batch_size = 100

learning_rate = 0.1



train_loss_list = []

train_acc_list = []

test_acc_list = []



iter_per_epoch = max(train_size / batch_size, 1)



for i in range(iters_num):

    batch_mask = np.random.choice(train_size, batch_size)

    x_batch = x_train[batch_mask]

    t_batch = t_train[batch_mask]

    

    # 梯度

    #grad = network.numerical_gradient(x_batch, t_batch)

    grad = network.gradient(x_batch, t_batch)

    

    # 更新

    for key in ('W1', 'b1', 'W2', 'b2'):

        network.params[key] -= learning_rate * grad[key]

    

    loss = network.loss(x_batch, t_batch)

    train_loss_list.append(loss)

    

    if i % iter_per_epoch == 0:

        train_acc = network.accuracy(x_train, t_train)

        test_acc = network.accuracy(x_test, t_test)

        train_acc_list.append(train_acc)

        test_acc_list.append(test_acc)

        print(train_acc, test_acc)
class SGD:

    def __init__(self, lr=0.01):

        self.lr = lr

        

    def update(self, params, grads):

        for key in params.keys():

            params[key] -= self.lr * grads[key]
class Momentum:



    """Momentum SGD"""



    def __init__(self, lr=0.01, momentum=0.9):

        self.lr = lr

        self.momentum = momentum

        self.v = None

        

    def update(self, params, grads):

        if self.v is None:

            self.v = {}

            for key, val in params.items():                                

                self.v[key] = np.zeros_like(val)

                

        for key in params.keys():

            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] 

            params[key] += self.v[key]
# AdaGrad实现

class AdaGrad:



    """AdaGrad"""



    def __init__(self, lr=0.01):

        self.lr = lr

        self.h = None

        

    def update(self, params, grads):

        if self.h is None:

            self.h = {}

            for key, val in params.items():

                self.h[key] = np.zeros_like(val)

            

        for key in params.keys():

            self.h[key] += grads[key] * grads[key]

            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
# Adam实现

class Adam:



    """Adam (http://arxiv.org/abs/1412.6980v8)"""



    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):

        self.lr = lr

        self.beta1 = beta1

        self.beta2 = beta2

        self.iter = 0

        self.m = None

        self.v = None

        

    def update(self, params, grads):

        if self.m is None:

            self.m, self.v = {}, {}

            for key, val in params.items():

                self.m[key] = np.zeros_like(val)

                self.v[key] = np.zeros_like(val)

        

        self.iter += 1

        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         

        

        for key in params.keys():

            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]

            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)

            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])

            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

            

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            

            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias

            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias

            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)
# coding: utf-8

import os

import sys

import matplotlib.pyplot as plt

from dataset.mnist import load_mnist

from common.util import smooth_curve

from common.multi_layer_net import MultiLayerNet

from common.optimizer import *





# 0:读入MNIST数据==========

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)



train_size = x_train.shape[0]

batch_size = 128

max_iterations = 2000





# 1:进行实验的设置==========

optimizers = {}

optimizers['SGD'] = SGD()

optimizers['Momentum'] = Momentum()

optimizers['AdaGrad'] = AdaGrad()

optimizers['Adam'] = Adam()

#optimizers['RMSprop'] = RMSprop()



networks = {}

train_loss = {}

for key in optimizers.keys():

    networks[key] = MultiLayerNet(

        input_size=784, hidden_size_list=[100, 100, 100, 100],

        output_size=10)

    train_loss[key] = []    





# 2:开始训练==========

for i in range(max_iterations):

    batch_mask = np.random.choice(train_size, batch_size)

    x_batch = x_train[batch_mask]

    t_batch = t_train[batch_mask]

    

    for key in optimizers.keys():

        grads = networks[key].gradient(x_batch, t_batch)

        optimizers[key].update(networks[key].params, grads)

    

        loss = networks[key].loss(x_batch, t_batch)

        train_loss[key].append(loss)

    

    if i % 100 == 0:

        print( "===========" + "iteration:" + str(i) + "===========")

        for key in optimizers.keys():

            loss = networks[key].loss(x_batch, t_batch)

            print(key + ":" + str(loss))





# 3.绘制图形==========

markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}

x = np.arange(max_iterations)

for key in optimizers.keys():

    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)

plt.xlabel("iterations")

plt.ylabel("loss")

plt.ylim(0, 1)

plt.legend()

plt.show()
# coding: utf-8

import numpy as np

import matplotlib.pyplot as plt





def sigmoid(x):

    return 1 / (1 + np.exp(-x))





def ReLU(x):

    return np.maximum(0, x)





def tanh(x):

    return np.tanh(x)

    

input_data = np.random.randn(1000, 100)  # 1000个数据

node_num = 100  # 各隐藏层的节点（神经元）数

hidden_layer_size = 5  # 隐藏层有5层

activations = {}  # 激活值的结果保存在这里



x = input_data



for i in range(hidden_layer_size):

    if i != 0:

        x = activations[i-1]



    # 改变初始值进行实验！

    w = np.random.randn(node_num, node_num) * 1

    # w = np.random.randn(node_num, node_num) * 0.01

    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)

    # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)





    a = np.dot(x, w)





    # 将激活函数的种类也改变，来进行实验！

    z = sigmoid(a)

    # z = ReLU(a)

    # z = tanh(a)



    activations[i] = z



# 绘制直方图

for i, a in activations.items():

    plt.subplot(1, len(activations), i+1)

    plt.title(str(i+1) + "-layer")

    if i != 0: plt.yticks([], [])

    # plt.xlim(0.1, 1)

    # plt.ylim(0, 7000)

    plt.hist(a.flatten(), 30, range=(0,1))

plt.show()

# coding: utf-8

import numpy as np

import matplotlib.pyplot as plt





def sigmoid(x):

    return 1 / (1 + np.exp(-x))





def ReLU(x):

    return np.maximum(0, x)





def tanh(x):

    return np.tanh(x)

    

input_data = np.random.randn(1000, 100)  # 1000个数据

node_num = 100  # 各隐藏层的节点（神经元）数

hidden_layer_size = 5  # 隐藏层有5层

activations = {}  # 激活值的结果保存在这里



x = input_data



for i in range(hidden_layer_size):

    if i != 0:

        x = activations[i-1]



    # 改变初始值进行实验！

#     w = np.random.randn(node_num, node_num) * 1

    w = np.random.randn(node_num, node_num) * 0.01

    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)

    # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)





    a = np.dot(x, w)





    # 将激活函数的种类也改变，来进行实验！

    z = sigmoid(a)

    # z = ReLU(a)

    # z = tanh(a)



    activations[i] = z



# 绘制直方图

for i, a in activations.items():

    plt.subplot(1, len(activations), i+1)

    plt.title(str(i+1) + "-layer")

    if i != 0: plt.yticks([], [])

    # plt.xlim(0.1, 1)

    # plt.ylim(0, 7000)

    plt.hist(a.flatten(), 30, range=(0,1))

plt.show()

# coding: utf-8

import numpy as np

import matplotlib.pyplot as plt





def sigmoid(x):

    return 1 / (1 + np.exp(-x))

    

input_data = np.random.randn(1000, 100)  # 1000个数据

node_num = 100  # 各隐藏层的节点（神经元）数

hidden_layer_size = 5  # 隐藏层有5层

activations = {}  # 激活值的结果保存在这里



x = input_data



for i in range(hidden_layer_size):

    if i != 0:

        x = activations[i-1]



    # 改变初始值进行实验！

    w = np.random.randn(node_num, node_num) / np.sqrt(node_num)



    a = np.dot(x, w)



    # 将激活函数的种类也改变，来进行实验！

    z = sigmoid(a)



    activations[i] = z



# 绘制直方图

for i, a in activations.items():

    plt.subplot(1, len(activations), i+1)

    plt.title(str(i+1) + "-layer")

    if i != 0: plt.yticks([], [])

    # plt.xlim(0.1, 1)

    # plt.ylim(0, 7000)

    plt.hist(a.flatten(), 30, range=(0,1))

plt.show()
#基于MNIST数据集的权重初始值的比较

# coding: utf-8

import os

import sys

import numpy as np

import matplotlib.pyplot as plt

from dataset.mnist import load_mnist

from common.util import smooth_curve

from common.multi_layer_net import MultiLayerNet

from common.optimizer import SGD





# 0:读入MNIST数据==========

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)



train_size = x_train.shape[0]

batch_size = 128

max_iterations = 2000





# 1:进行实验的设置==========

weight_init_types = {'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}

optimizer = SGD(lr=0.01)



networks = {}

train_loss = {}

for key, weight_type in weight_init_types.items():

    networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100],

                                  output_size=10, weight_init_std=weight_type)

    train_loss[key] = []





# 2:开始训练==========

for i in range(max_iterations):

    batch_mask = np.random.choice(train_size, batch_size)

    x_batch = x_train[batch_mask]

    t_batch = t_train[batch_mask]

    

    for key in weight_init_types.keys():

        grads = networks[key].gradient(x_batch, t_batch)

        optimizer.update(networks[key].params, grads)

    

        loss = networks[key].loss(x_batch, t_batch)

        train_loss[key].append(loss)

    

    if i % 100 == 0:

        print("===========" + "iteration:" + str(i) + "===========")

        for key in weight_init_types.keys():

            loss = networks[key].loss(x_batch, t_batch)

            print(key + ":" + str(loss))





# 3.绘制图形==========

markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}

x = np.arange(max_iterations)

for key in weight_init_types.keys():

    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)

plt.xlabel("iterations")

plt.ylabel("loss")

plt.ylim(0, 2.5)

plt.legend()

plt.show()