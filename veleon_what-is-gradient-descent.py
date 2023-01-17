from IPython.display import Image

from IPython.core.display import HTML 

Image(url= "https://cdn-images-1.medium.com/max/1600/0*rBQI7uBhBKE8KT-X.png")
import numpy as np

import matplotlib.pyplot as plt

np.random.seed(42)

X = np.random.rand(100,1)

y = 2 + X+np.random.randn(100,1)/7.5

y_no_noise = 2 + X
plt.plot(X, y, 'ro')

plt.plot(X, y_no_noise, 'go')

plt.show()
def calculateGradientVector(X, y, theta):

    X_b = np.c_[np.ones((len(X),1)), X] # concatenate a 1 to each instance (because we dont have x_0 in X)

    return (2/len(X))*X_b.T.dot(X_b.dot(theta)-y)



def batchGradientDescent(X, y, eta, iterations):

    np.random.seed(42)

    theta = np.random.randn(2,1)

    for i in range(iterations):

        gradientVector = calculateGradientVector(X, y, theta)

        theta = theta - eta * gradientVector

    return theta



batchGradientDescent(X,y,0.1,1000)
def predictY(x, theta): # predicts a single y value

    return theta[0]+theta[1]*x



def getLearningRatePlot(X, y, eta, iterations):

    plt.plot(X, y, 'ro')

    for i in range(iterations):

        theta = batchGradientDescent(X, y, eta, i)

        if i is 0:

            plt.plot([0, 1],[predictY(0, theta), predictY(1, theta)], 'r--')

        elif i is iterations-1:

            plt.plot([0, 1],[predictY(0, theta), predictY(1, theta)], 'g-' , linewidth=3)

        else:

            plt.plot([0, 1],[predictY(0, theta), predictY(1, theta)], 'b-')

    plt.xlabel('$X_1$', fontsize=20)

    plt.title("$\eta = {}$ for ${}$ iterations".format(eta, iterations), fontsize=20)

    plt.axis([0, 1, 0, 4])
plt.figure(figsize=(20,4))

plt.subplot(131); getLearningRatePlot(X, y, 0.02, 10)

plt.ylabel("$y$", rotation=0, fontsize=18)

plt.subplot(132); getLearningRatePlot(X, y, 0.1, 10)

plt.subplot(133); getLearningRatePlot(X, y, 0.8, 10)
plt.figure(figsize=(20,4))

plt.subplot(131); getLearningRatePlot(X, y, 0.02, 100)

plt.ylabel("$y$", rotation=0, fontsize=18)

plt.subplot(132); getLearningRatePlot(X, y, 0.1, 100)

plt.subplot(133); getLearningRatePlot(X, y, 0.8, 100)
def learningSchedule(t, t0=5, t1=50):

    #t0 and t1 define your starting eta (5/50 = 0.1) and the growth rate (1/11 = 0.090 but 5/51 = 0.098)

    return t0 / (t+t1)



def stochasticGradientDescent(X, y, epochs, t0=5, t1=50):

    np.random.seed(42)

    theta = np.random.randn(2,1)

    for epoch in range(epochs):

        for i in range(len(X)):

            random_iteration = np.random.randint(len(X))

            x_i = X[random_iteration:random_iteration+1]

            y_i = y[random_iteration:random_iteration+1]

            gradientVector = calculateGradientVector(x_i, y_i, theta)

            eta = learningSchedule(epoch*len(X)+i, t0, t1)

            theta = theta - eta*gradientVector

    return theta



stochasticGradientDescent(X, y, 1000)
def plotStochastic(X,y,iterations, t0=5, t1=50): 

    plt.plot(X, y, 'ro')

    for i in range(iterations):

        theta = stochasticGradientDescent(X, y, i, t0, t1)

        if i is 0:

            plt.plot([0, 1],[predictY(0, theta), predictY(1, theta)], 'r--')

        elif i is iterations-1:

            plt.plot([0, 1],[predictY(0, theta), predictY(1, theta)], 'g-' , linewidth=3)

        else:

            plt.plot([0, 1],[predictY(0, theta), predictY(1, theta)], 'b-')

    plt.xlabel('$X_1$', fontsize=20)

    plt.title("SGD for ${}$ iterations with schedule {}/{}".format(iterations,t0,t1), fontsize=15)

    plt.axis([0, 1, 0, 4])
plt.figure(figsize=(20,4))

plt.subplot(131); plotStochastic(X, y, 10)

plt.ylabel("$y$", rotation=0, fontsize=18)

plt.subplot(132); plotStochastic(X, y, 10, 1, 100)

plt.subplot(133); plotStochastic(X, y, 10, 1, 200)
def miniBatchGradientDescent(X, y, epochs, batchSize, t0=5, t1=50):

    np.random.seed(42)

    theta = np.random.randn(2,1)

    for epoch in range(epochs):

        for i in range(len(X)):

            x_i = np.zeros((batchSize,1))

            y_i = np.zeros((batchSize,1))

            for b in range(batchSize):

                random_iteration = np.random.randint(len(X))

                x_i[b] = X[random_iteration]

                y_i[b] = y[random_iteration]

            gradientVector = calculateGradientVector(x_i, y_i, theta)

            eta = learningSchedule(epoch*len(X)+i, t0, t1)

            theta = theta - eta*gradientVector

    return theta



miniBatchGradientDescent(X, y, 1000, 20)
def plotMBGD(X,y,iterations, batchSize): 

    plt.plot(X, y, 'ro')

    for i in range(iterations):

        theta = miniBatchGradientDescent(X, y, i, batchSize, t0=1, t1=100)

        if i is 0:

            plt.plot([0, 1],[predictY(0, theta), predictY(1, theta)], 'r--')

        elif i is iterations-1:

            plt.plot([0, 1],[predictY(0, theta), predictY(1, theta)], 'g-' , linewidth=3)

        else:

            plt.plot([0, 1],[predictY(0, theta), predictY(1, theta)], 'b-')

    plt.xlabel('$X_1$', fontsize=20)

    plt.title("MBGD for ${}$ iterations with batchsize {}".format(iterations,batchSize), fontsize=15)

    plt.axis([0, 1, 0, 4])

    

plt.figure(figsize=(20,4))

plt.subplot(121); plotStochastic(X, y, 10, 1, 100)

plt.ylabel("$y$", rotation=0, fontsize=18)

plt.subplot(122); plotMBGD(X, y, 10, 10)