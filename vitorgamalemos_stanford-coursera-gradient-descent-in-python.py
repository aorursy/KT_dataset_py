import numpy as np

import matplotlib.pyplot as plt



def mse(y_pred, y):

    error = np.sum(np.subtract(y_pred, y)**2)

    return 1 / (2 * len(y_pred)) * error
y_pred = np.array([[1, 0, 1, 0, 0],

                   [1, 0, 1, 0, 0],

                   [1, 0, 1, 0, 0],

                   [1, 0, 1, 0, 0]])

y = np.array([[1, 1, 0, 0, 1], 

              [1, 0, 0, 0, 1],

              [1, 0, 1, 0, 1],

              [1, 0, 1, 0, 0]])



err = []

for pred, target in zip(y_pred, y):

    err.append(mse(pred, target))
plt.title("MSE")

plt.plot(err, 'r--', label="error")

plt.xlabel("i")

plt.ylabel("error")

plt.grid()

plt.legend()
plt.plot(y[0], "--ro", label="real y")

plt.plot(y_pred[0], "--ro", label="predict y", color="blue")

plt.grid()

plt.legend()
plt.plot(y[1], "--ro", label="real y")

plt.plot(y_pred[1], "--ro", label="predict y", color="blue")

plt.grid()

plt.legend()
def cost_function(y, y_pred):

    return (1 / len(y)) * sum([k**2 for k in (y - y_pred)])



def update_w(W, wb, b, bd, learning_rate):

    return W - learning_rate * wb, b - learning_rate * bd



def gradient_descent(X, y, learning_rate=0.05, 

                           max_epochs=10,

                           W=0.01, b=0.01):

    

    cost_, epoch_ = list(), list()

    for i in range(max_epochs):

        y_pred = W * X + b

        cost = cost_function(y, y_pred)

        wd = -(2 / len(X)) * sum(X *(y - y_pred))

        bd = -(2 / len(X)) * sum(y - y_pred)

        

        W, b = update_w(W, wd, b, bd, learning_rate)

        

        cost_.append(cost)

        epoch_.append(i)

        

    return cost_, epoch_



X = np.array([1, 2, 3, 4, 5])

y = np.array([5, 7, 9, 11, 13])



cost, epoch = gradient_descent(X, y)

plt.plot(epoch, cost, "--ro", label="error")

plt.grid()

plt.legend()

plt.xlabel("Epochs")

plt.ylabel("Cost")

plt.title(" Gradient descent")