import numpy as np
import matplotlib.pyplot as plt

import sklearn.datasets
from sklearn.svm import SVC
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
def plot_data(X, y):
    plt.scatter(X[:,0], X[:,1], s=40, c=y)

print(X.shape, y.shape)
plot_data(X, y)
svc = SVC(kernel='linear')
svc.fit(X, y)
def plot_decision_boundary(X, y, clf, steps=100):
    min_x, min_y = np.min(X, axis=0)
    max_x, max_y = np.max(X, axis=0)
    x_range = np.linspace(min_x, max_x, num=steps)
    y_range = np.linspace(min_y, max_y, num=steps)
    boundary = np.zeros((steps, 2))
    for i, x in enumerate(x_range):
        k = clf.predict([[x, min_y]])[0]
        for y in y_range:
            if k != clf.predict([[x, y]])[0]:
                boundary[i] = [x, y]
                break
    plt.plot(boundary[:,0], boundary[:,1], '-')
plot_data(X, y)
plot_decision_boundary(X, y, svc)
def relu(x):
    x[x < 0] = 0
    return x

softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

class NN:
    def __init__(self, dim_in, dim_H, dim_out):
        self.Wh = np.zeros((dim_in, dim_H))
        self.bh = np.zeros((1, dim_H))
        self.Wo = np.zeros((dim_H, dim_out))
        self.bo = np.zeros((1, dim_out))
        self.h = self.o = None
    
    def forward(self, X):
        self.h = relu(X.dot(self.Wh) + self.bh)
        self.o = softmax(self.h.dot(self.Wo) + self.bo)
        return self.o
nn = NN(2, 4, 2)
nn.forward(np.array([[0, 0]]))
def backprop(self):
    pass

NN.backprop = backprop
def train(self, X):
    pass

NN.train = train