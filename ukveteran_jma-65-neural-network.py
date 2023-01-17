import numpy as np

import matplotlib.pyplot as plt



np.random.seed(1)



def sigmoid(x):

    return 1 / (1 + np.exp(-x))



def softmax(A):

    expA = np.exp(A)

    return expA / expA.sum(axis=1, keepdims=True)



# a 3 x 4 x 3 network example

s = 5 # five samples

n = 3 # three features per sample

d = 4 # four nodes in the hidden layer

c = 3 # three classes to predict



X = np.random.randn(s, n)



W = np.random.randn(n, d)

b = np.random.randn(d)



Z = X.dot(W) + b

A = sigmoid(Z)



W2 = np.random.randn(d, c)

b2 = np.random.randn(c)



Z2 = A.dot(W2) + b2

Y = softmax(Z2)



# generate three Gaussian clouds each holding 500 points

X1 = np.random.randn(500, 2) + np.array([0, -2])

X2 = np.random.randn(500, 2) + np.array([2, 2])

X3 = np.random.randn(500, 2) + np.array([-2, 2])



# put them all in a big matrix

X = np.vstack([X1, X2, X3])



# generate the one-hot-encodings

labels = np.array([0]*500 + [1]*500 + [2]*500)

T = np.zeros((1500, 3))

for i in range(1500):

    T[i, labels[i]] = 1



# visualize the data

plt.scatter(X[:,0], X[:,1], c=labels, s=10, alpha=0.5)

plt.show()





samples = X.shape[0] # 1500 samples

features = X.shape[1] # 2 features

hidden_nodes = 5

classes = 3



# randomly initialize weights

W1 = np.random.randn(features, hidden_nodes)

b1 = np.random.randn(hidden_nodes)

W2 = np.random.randn(hidden_nodes, classes)

b2 = np.random.randn(classes)



alpha = 10e-6

costs = []

for epoch in range(10000):

    # forward pass

    A = sigmoid(X.dot(W1) + b1) # A = sigma(Z)

    Y = softmax(A.dot(W2) + b2) # Y = softmax(Z2)



    # backward pass

    delta2 = Y - T

    delta1 = (delta2).dot(W2.T) * A * (1 - A)



    W2 -= alpha * A.T.dot(delta2)

    b2 -= alpha * (delta2).sum(axis=0)



    W1 -= alpha * X.T.dot(delta1)

    b1 -= alpha * (delta1).sum(axis=0)



    # save loss function values across training iterations

    if epoch % 100 == 0:

        loss = np.sum(-T^2 * np.log(Y))

        print('Loss function value: ', loss)

        costs.append(loss)



        

plt.plot(costs)

plt.show()
