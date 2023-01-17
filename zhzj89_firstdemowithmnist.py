# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



print("hello kaggle!")



# Any results you write to the current directory are saved as output.
x_data = np.float32(np.random.rand(2,100))

y_data = np.dot([0.100,0.200], x_data) + 0.300



#print("x_data", x_data)

#print("y_data", y_data)



b = tf.Variable(tf.zeros([1]))

W = tf.Variable(tf.random_uniform([1,2],-1.0,1.0))

y = tf.matmul(W, x_data) + b



#print("y:", y)

#print("W:", W)

#print("b:", b)



loss = tf.reduce_mean(tf.square(y-y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)

train = optimizer.minimize(loss)

#print("loss: ", loss)

#print("optimizer: ", optimizer)

#print("train: ", train)



init = tf.global_variables_initializer()

#print("init:", init)



sess = tf.Session()

sess.run(init)

for step in range(0,301):

    sess.run(train)

    if step % 20 == 0:

        print(step,sess.run(W), sess.run(b),sess.run(loss))

    




# 使用 NumPy 生成假数据(phony data), 总共 100 个点.

x_data = np.float32(np.random.rand(2, 100)) # 随机输入

y_data = np.dot([0.100, 0.200], x_data) + 0.300



# 构造一个线性模型

# 

b = tf.Variable(tf.zeros([1]))

W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))

y = tf.matmul(W, x_data) + b



# 最小化方差

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)

train = optimizer.minimize(loss)



# 初始化变量

init = tf.global_variables_initializer()



# 启动图 (graph)

sess = tf.Session()

sess.run(init)



# 拟合平面

for step in range(0, 201):

    sess.run(train)

    if step % 20 == 0:

        print("", step, sess.run(W), sess.run(b))



# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]