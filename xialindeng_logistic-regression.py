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



def sigmoid(inX):

    return 1.0/(1+np.exp(-inX))

 

def plotloss(loss_list):

    n = len(loss_list)

    plt.xlabel("iteration num")

    plt.ylabel("loss")

    plt.scatter(range(1, n+1), loss_list)

    plt.show()



def my_logistic_regression(x_total, y_total):

    loss_list = []

    m,n=np.shape(x_total)

    w=np.zeros((n,1))

    b=0

    for i in range(n_iterations):

        A=sigmoid(np.dot(x_total,w)+b)

        loss=-(np.sum(y_total*np.log(A)+(1-y_total)*np.log(1-A)))/m

        dz=A-y_total

        dw=(np.dot(x_total.T,dz))/m

        db=(np.sum(dz))/m

        w=w-learning_rate*dw

        b=b-learning_rate*db

        loss_list.append(loss)

    plotloss(loss_list)

    y_pre=np.zeros((m,1))

    A=sigmoid(np.dot(x_total,w)+b)

    for i in range(m):

        if A[i,0]>0.5:

            y_pre[i,0]=1

        else:

            y_pre[i,0]=0

    plot_x = np.linspace(-1.0, 1.0, 100)

    plot_y = - (w[0][0] * plot_x ) / w[0][1]+b

    plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

    plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

    plt.plot(plot_x, plot_y, c='g')

    plt.show()        

    return y_pred



y_pred = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())