# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

from random import seed

import matplotlib.pyplot as plt

import numpy as np

seed(1)
class GD:



    def __init__(self, d, lr, iterations):

        self.params = np.random.random_sample((d+1, 1))

        self.lr = lr

        self.iterations = iterations



    def fit(self, X, Y):

        w = self.params

        N = X.shape[0]

        for i in range(self.iterations):

            yhat = np.matmul(X, w)

            grad = -(1 / N) * 2 * np.matmul((Y - yhat).T, X)

            w = w - self.lr * grad.reshape((-1, 1))

        return w

    

    def fit_reg(self, X, Y, hp):

        w = self.params

        N = X.shape[0]

        for i in range(self.iterations):

            yhat = np.matmul(X, w)

            grad = -(1 / N) * 2 * np.matmul((Y - yhat).T, X)

            w = w - self.lr * (grad.reshape((-1, 1)) + 2 * hp * w)

        return w
class SGD:



    def __init__(self, d, lr, iterations):

        self.params = np.random.random_sample((d+1,1))

        self.lr = lr

        self.iterations = iterations



    def fit(self, X, Y):

        w = self.params

        N = X.shape[0]

        if self.iterations >= N:

            self.iterations = N

        for i in range(self.iterations):

            Xentry = X[i,:]

            yhat = np.matmul(Xentry, w)

            grad = -1 * 2 * (Y[i, 0] - yhat) * Xentry.reshape((-1, 1))

            w = w - self.lr * grad

        return w
class MinibatchedSGD:



    def __init__(self, d, lr, iterations, batchsize):

        self.params = np.random.random_sample((d+1,1))

        self.lr = lr

        self.iterations = iterations

        self.batchsize = batchsize



    def fit(self, X, Y):

        w = self.params

        N = X.shape[0]

        for i in range(self.iterations):

            batchNum = int(N / self.batchsize) + 1

            id = np.arange(len(X))

            np.random.shuffle(id)

            for k in range(batchNum):

                Xmini = X[k * self.batchsize: (k + 1) * self.batchsize,:]

                Ymini = Y[k * self.batchsize: (k + 1) * self.batchsize,:].reshape((-1,1))

                yhat = np.matmul(Xmini, w)

                grad = -(1 / self.batchsize) * 2 * np.matmul((Ymini - yhat).T, Xmini)

                w = w - self.lr * grad.reshape((-1, 1))

        return w
'''

function getdata

input: N number of entries

       miu mean value of normal distribution (the distribution of Z)

       var variance of normal distribution

output: X generated data points

        Y corresponding value computed by true distribution

'''

def getdata(N, miu, var, d):

    X = np.random.random_sample((N,1))

    Z = np.random.normal(miu, var, X.shape)

    Y = np.cos(2 * np.pi * X) + Z

    return convert2poly(X, d), Y
'''

function convert2poly

converts input data to polynomial form

input: X data entries

output: d degree of polynomial

'''

def convert2poly(X, d):

    Xpoly = np.ones(X.shape)

    for ite in range(1,d+1):

        buf = X ** ite

        Xpoly = np.append(Xpoly, buf, axis=1)

    return Xpoly
'''

function fitData

input: X, Y training data

       optimizer one of SGD, GD, minibatched SGD object

output: w weights of fitted model

'''

def fitData(X, Y, optimizer):

    w = optimizer.fit(X, Y)

    return w
def fitData_reg(X, Y, optimizer, hp):

    w = optimizer.fit_reg(X, Y, hp)

    return w
'''

function getMSE

input: two column vectors Y and yhat

output: the mean square error of these two vectors

'''

def getMSE(Y, yhat):

    N = len(Y)

    return np.sum((Y - yhat) ** 2) / N
def experiment(N, d, var, lr, iterations, M, hp):

    train_err = 0

    test_err = 0

    w_ave = 0

    test_size = 1000

    # yhat_save = np.random.rand(M, test_size, 1)

    w_save = np.random.rand(M, d + 1, 1)

    for j in range(M):

        Xtrain, Ytrain = getdata(N, 0, var, d) # draw data

        Xtest, Ytest = getdata(test_size, 0, var, d)

        # optimizer = MinibatchedSGD(d=d, lr=lr, iterations=iterations, batchsize=batchsize)

        optimizer = GD(d=d, lr=lr, iterations=iterations)

        # optimizer = SGD(d=d, lr=lr, iterations=iterations)

        w = fitData_reg(Xtrain, Ytrain, optimizer, hp)

        # w = fitData(Xtrain, Ytrain, optimizer)

        w_save[j, :] = w

        yhat = np.matmul(Xtrain, w)

        ypred = np.matmul(Xtest, w)

        # yhat_save[j, :] = ypred

        # plt.scatter(Xtest[:,1], ypred)

        # plt.scatter(Xtrain[:,1], yhat)

        train_errj = getMSE(Ytrain, yhat)

        train_err += train_errj

        test_errj = getMSE(Ytest, ypred)

        test_err += test_errj

        w_ave += w

    train_err = train_err / M

    test_err = test_err / M

    w_ave = w_ave / M

    #compute bias

    X_true, Y_true = getdata(test_size, 0, var, d)

    # plt.scatter(X_true[:,1], Y_true)

    y_ave = np.matmul(X_true, w_ave)

    bias = getMSE(y_ave, Y_true)

    # plt.scatter(X_true[:,1], y_ave)

    # plt.show()

    xaxis = np.arange(0, 1, 1/1000).reshape((-1,1))

    xpoly = convert2poly(xaxis, d)

    yaxis = np.matmul(xpoly, w_save)

    y_var_ave = np.matmul(xpoly, w_ave)

    variance = 0

    for l in range(M):

        v = getMSE(yaxis[l, :], y_var_ave)

        variance += v

    variance = variance / M

    # variance = np.sum((yaxis - y_var_ave) ** 2) / (yaxis.shape[0] * yaxis[1] * yaxis[2])

    return w_ave, train_err, test_err, bias, variance

hp = 0.01

d = np.arange(0,21)

N = np.array([2, 5, 10, 20, 50, 100, 200])

var = np.array([1, 0.1, 0.01])

lr = 0.1

iterations = 5000

# best result so far var = 1 N = 50 lr = 0.2 iteration = 2000

# best result with d = 20 lr = 0.2 iteration = 2000 N = 30 var = 1

batchsize = 6

M = 50

var = 1

N = 30

train_err = np.zeros(len(d))

test_err = np.zeros(len(d))

bias = np.zeros(len(d))

variance = np.zeros(len(d))

for di in d:

    w_ave, train_err[di], test_err[di], bias[di], variance[di] = experiment(N, di, var, lr, iterations, M, hp)

# print("w_ave", w_ave, "\ntrain_error", train_err, "\ntest_error", test_err, "\nbias", bias, "\nvariance", variance, "\nsum", variance + bias)

plt.figure("training testing error")

plt.plot(d, train_err, label="training error")

plt.plot(d, test_err, label="testing error")

# plt.xticks([i for i in range(0,21)])

plt.xlabel("d")

plt.legend()

plt.show()



plt.figure("training testing error")

plt.plot(d, bias, label="bias")

plt.plot(d, variance, label="variance")

# plt.xticks([i for i in range(0,21)])

plt.xlabel("d")

plt.legend()

plt.show()

# x = X_train[:,1]

# plt.scatter(x, yhat)

# plt.show()

hp = 0.01

d1 = 2

d2 = 20

N = np.array([10, 20, 50, 100, 150, 200, 300, 400])

var = 1

lr = 0.1

iterations = 2000

M = 50

train_errd1 = np.zeros(len(N))

test_errd1 = np.zeros(len(N))

biasd1 = np.zeros(len(N))

varianced1 = np.zeros(len(N))

for ni in range(len(N)):

    w_ave, train_errd1[ni], test_errd1[ni], biasd1[ni], varianced1[ni] = experiment(N[ni], d1, var, lr, iterations, M, hp)

print(biasd1)

print(varianced1)

plt.figure("training testing error of simple model")

plt.plot(N, train_errd1, label="training error")

plt.plot(N, test_errd1, label="testing error")

# plt.xticks([i for i in range(0,21)])

plt.xlabel("N")

plt.ylim(0, 3)

plt.legend()

plt.show()



# plt.figure("bias variance")

# plt.plot(N, varianced1, label="variance", c='r')

# plt.plot(N, biasd1, label="bias", c='b')

# plt.plot(N, biasd1 + varianced1, label="bias + variance", c='g')

# plt.plot(N,test_errd1, label="test error", c='k')

# # plt.xticks([i for i in range(0,21)])

# plt.xlabel("N")

# plt.legend()

# plt.show()



train_errd2 = np.zeros(len(N))

test_errd2 = np.zeros(len(N))

biasd2 = np.zeros(len(N))

varianced2 = np.zeros(len(N))

for ni in range(len(N)):

    w_ave, train_errd2[ni], test_errd2[ni], biasd2[ni], varianced2[ni] = experiment(N[ni], d2, var, lr, iterations, M, hp)

plt.figure("training testing error of complex model")

plt.plot(N, train_errd2, label="training error")

plt.plot(N, test_errd2, label="testing error")

# plt.xticks([i for i in range(0,21)])

plt.xlabel("N")

plt.ylim(0, 3)

plt.legend()

plt.show()
