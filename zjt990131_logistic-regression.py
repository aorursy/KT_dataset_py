import numpy as np

import matplotlib.pyplot as plt



lines = np.loadtxt('../input/data.csv', delimiter=',', dtype='str')

x_total = lines[:, 1:3].astype('float')

y_total = lines[:, 3].astype('float')



pos_index = np.where(y_total == 1)

neg_index = np.where(y_total == 0)

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.show()

print('Data set size:', x_total.shape[0])
from sklearn import linear_model



lr_clf = linear_model.LogisticRegression()

lr_clf.fit(x_total, y_total)

print(lr_clf.coef_[0])

print(lr_clf.intercept_)



y_pred = lr_clf.predict(x_total)

print('accuracy:',(y_pred == y_total).mean())



plot_x = np.linspace(-1.0, 1.0, 100)

plot_y = - (lr_clf.coef_[0][0] * plot_x + lr_clf.intercept_[0]) / lr_clf.coef_[0][1]

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.plot(plot_x, plot_y, c='g')

plt.show()
# 1. finish function my_logistic_regression;

# 2. draw a training curve (the x-axis represents the number of training iterations, and the y-axis represents the training loss for each round);

# 3. draw a pic to show the result of logistic regression (just like the pic in section 2);



n_iterations = 2000

learning_rate = 0.1

loss_list = []

def sigmoid(x):#定义sigmoid函数，方便调用

    return(1.0/(1+np.exp(-x)))

def Loss(y,p):#定义loss函数，方便调用

    return ((-y*np.log(p)-(1-y)*np.log(1-p)).mean())

def my_logistic_regression(x_total, y_total):

    # TODO

    n=x_total.shape[0]#训练数据的个数200

    m=x_total.shape[1]#数据维度2

    w=np.ones(m)

    b=0.0

    for i in range(n_iterations):

        y_pred=sigmoid(np.dot(x_total,w)+b)

        grad_w=np.dot((y_pred-y_total),x_total)/n#利用矩阵的点乘操作实现w梯度的计算

        w-=grad_w*learning_rate

        grad_b=(y_pred-y_total).mean()

        b-=grad_b*learning_rate#更新参数

        loss=Loss(y_total,y_pred)

        loss_list.append(loss)

    for j in range(n):

        if y_pred[j]>=0.5:

            y_pred[j]=1

        else:

            y_pred[j]=0

    return y_pred,w,b,loss_list#为了方便观察返回了四个值，分别是拟合得到的w,b以及整个过程中的loss值

y_pred,w,b,loss=my_logistic_regression(x_total, y_total)

print(w)

print(b)

print('accuracy:',(y_pred == y_total).mean())



pos_index = np.where(y_total == 1)

neg_index = np.where(y_total == 0)

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plot_x=np.linspace(-1.0, 1.0, 100)

plot_y=-(w[0]*plot_x+b)/w[1]

plt.plot(plot_x, plot_y, c='g')

plt.show()

print('Data set size:', x_total.shape[0])



plot_xx=np.array(range(1,2001))

plot_yy=np.array(loss)

plt.plot(plot_xx,plot_yy,c='b')

plt.show()

print(loss)