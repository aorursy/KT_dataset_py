import numpy as np
import matplotlib.pyplot as plt

lines = np.loadtxt('../input/data.csv', delimiter=',', dtype='str')
#print(lines)
x_total = lines[:, 1:3].astype('float')
y_total = lines[:, 3].astype('float')
#print(x_total)
#print(y_total)
pos_index = np.where(y_total == 1)
neg_index = np.where(y_total == 0)
plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')
plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')
plt.show()
print('Data set size:', x_total.shape[0])
from sklearn import linear_model

lr_clf = linear_model.LogisticRegression()
lr_clf.fit(x_total, y_total)
#print(lr_clf.coef_)
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

from numpy import *
n_iterations = 2000
learning_rate = 0.1
loss_list = []
def sigmoid(x):
    a = 1 / (1 + np.exp(-x))
    return a
def prediction(w, Data):
    pred = []
    z = np.dot(w, Data)
    a = sigmoid(z)
    for i in range(0,len(a[0])):
        if (a[0][i] > 0.5): 
            pred.append(1)
        elif (a[0][i] <= 0.5):
            pred.append(0)
    return pred
def los(h, y):
    #return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    cross_entropy = -np.dot(y, np.log(h)) - np.dot((1 - y) , np.log(1 - h))
    return cross_entropy



def my_logistic_regression(x_total, y_total):
    # TODO
    w = np.random.randn(1,3) # create random weight
    
    Data = np.concatenate((x_total, np.ones((x_total.shape[0], 1))), axis = 1)
    
    #Data = Data.T # 3 * n after transpose
    for i in range(1, n_iterations):
        z = np.dot(w, Data.T)
        #y_pred = prediction(w, Data.T)
        h = sigmoid(z)
        #gradient = np.dot((h - y_total), Data) / y_total.size
        gradient = np.dot((h - y_total), Data) / y_total.size
        
        w = w - learning_rate * gradient
        
        z = np.dot(Data, w.T)
        h = sigmoid(z)
        loss = los(h, y_total)
        #print(loss) 
    
        loss_list.append(loss)
    domain =  np.linspace(-1,1,100)
    h_x = -(w[0,0]/w[0,1])*domain + (w[0,2]/w[0,1])
    plt.figure(1)
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)
    plt.sca(ax2)
    plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')
    plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')
    plt.plot(domain,h_x)
    plt.sca(ax1)
    x = range(1,n_iterations)
    plt.scatter(x, loss_list, marker='x', c='g')
    
    plt.show()
    
    return y_pred

y_pred = my_logistic_regression(x_total, y_total)
#print(y_pred)
print('accuracy:',(y_pred == y_total).mean())
