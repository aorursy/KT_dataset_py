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

import math

n_iterations = 2000

learning_rate = 0.1

loss_list = []



def my_logistic_regression(x_total, y_total):########助教你好，要求里让画两份图，因为subplot函数无法把这两张图一起很好的画出来，因此我在最下方新建了一行code用来画lost_list[]的图

    ceta0=0

    ceta=np.zeros((2,1))

    lastceta0=0

    lastceta=np.zeros((2,1))

    

    for i in range(n_iterations):

        ceta0=ceta0-learning_rate/100*np.sum(logisticfunction(ceta0,np.dot(x_total,lastceta))-y_total)

        ceta[0]=ceta[0]-learning_rate/100*np.sum(np.multiply(logisticfunction(lastceta0,np.dot(x_total,lastceta))-y_total,x_total[:,0]))

        ceta[1]=ceta[1]-learning_rate/100*np.sum(np.multiply(logisticfunction(lastceta0,np.dot(x_total,lastceta))-y_total,x_total[:,1]))

        lastceta0=ceta0

        lastceta=ceta

        cetax=ceta0+np.dot(x_total,ceta)

        loss_list.append(costfunction(cetax,y_total)/100)

    y_total=[]

    res=np.dot(x_total,ceta)

    for i in range(100):

        if res[i]+ceta0<0:

            y_total.append(0)

        else:

            y_total.append(1)

    

            

    plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

    plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')  

    plot_x = np.linspace(-1.0, 1.0, 100)

    plot_y = - (ceta[0] * plot_x + ceta0) / ceta[1]

    plt.plot(plot_x, plot_y, c='g') ###下方为了方便计算和代码编写，自己定义了两个函数



    return y_total   



def logisticfunction(ceta0,cetax):

    tmp=np.zeros((1,100))

    for i in range(100):

        tmp[0,i]=1./(1.+np.exp(-ceta0-cetax[i])) 

    return tmp



def costfunction(cetax,y):

    cost=0

    for i in range(100):

        if y[i]==1:

            cost=cost-np.log(1./(1.+np.exp(-cetax[i])))

        else:

            cost=cost-np.log(1.-1./(1.+np.exp(-cetax[i])))

    return cost

    







y_pred = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())    













plt.plot(range(len(loss_list)),(loss_list))