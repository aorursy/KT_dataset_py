import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



x = pd.read_csv('../input/air-predict/Train.csv').values

mean = np.mean(x, axis = 0)

standard = np.std(x, axis = 0)

X = (x-m)/s

Y = X[:, 5]

X = X[:, :5]



def hypothesis(X, thetha):

    y_ = thetha[0]

    n = X.shape[0] 

    for i in range(n):

        y_ += thetha[i+1]*X[i]

    return y_

def error(X, Y, thetha):

    error = 0.0

    m = X.shape[0]

    for i in range(m):

        y_ = hypothesis(X[i], thetha)

        error += (Y[i]-y_)**2

    return error/m

def gradient(X, Y, thetha):

    m, n = X.shape

    grad = np.zeros((n+1))

    for j in range(1, n+1):

        for i in range(m):

            y_ = hypothesis(X[i], thetha)

            grad[0] += (y_ - Y[i])

            grad[j] += (y_ - Y[i])*X[i][j-1]

    return grad/m

def graddescent(X, Y, rate = 0.05, num = 300):

    m, n = X.shape

    thetha = np.zeros(n+1)

    error_list= []

    for i in range(num):

        e = error(X, Y, thetha)

        error_list.append(e)

        grad = gradient(X, Y, thetha)

        for j in range(len(thetha)):

            thetha[j] = thetha[j] - rate*grad[j]

    return thetha, error_list
thetha, error_list = graddescent(X, Y)

print(thetha)

plt.plot(error_list)
xt = pd.read_csv('../input/air-predict/Test.csv').values

m= mean[:5]

std = standard[:5]

XT = (xt-m)/std

YT = []

l = XT.shape[0]

for i in range(l):

    y = hypothesis(XT[i], thetha)

    YT.append(y)

z = np.arange(l)

YT = np.array(YT)

YT = (YT*standard[5])+mean[5]

f = np.column_stack((z, YT))

df = pd.DataFrame(data = f, columns = ["Id", "target"])

print(df)

df.to_csv('OUTPUT.csv', index = False)