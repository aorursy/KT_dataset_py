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

x_total_all = np.hstack([x_total, np.ones([x_total.shape[0], 1])])

def sigmoid_function(w):
    result = 1/(1+np.exp(-w))
    return result

def my_logistic_regression(x_total, y_total):
    # TODO
    weight = np.zeros(3)
    for i in range(n_iterations):
        prob_predict = sigmoid_function(np.dot(x_total_all, weight))
        loss = (- y_total * np.log(prob_predict) - (1 - y_total) * (np.log(1 - prob_predict))).mean()
        loss_list.append(loss)
        grad_w = (x_total_all * np.tile((prob_predict - y_total).reshape([-1, 1]), 3)).mean(axis=0)
        weight = weight - learning_rate * grad_w
        
    y_total=np.where(np.dot(x_total_all,weight)>0,1,0)    
    return y_total

y_pred = my_logistic_regression(x_total, y_total)
print('accuracy:',(y_pred == y_total).mean())

x1=np.linspace(1,n_iterations,n_iterations)
y1=loss_list

x2 = np.linspace(-1.0, 1.0, 100)
y2 = - (lr_clf.coef_[0][0] * plot_x + lr_clf.intercept_[0]) / lr_clf.coef_[0][1]

fig1=plt.figure(2)
plt.subplot(211)
plt.plot(x1,y1,c='b')
plt.xlabel("iteration")
plt.ylabel("training loss")
plt.show()
plt.subplot(212)
plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')
plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')
plt.plot(x2, y2, c='g')
plt.show()