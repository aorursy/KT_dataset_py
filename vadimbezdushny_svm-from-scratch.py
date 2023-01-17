%matplotlib inline



import numpy as np

import matplotlib.pyplot as plt

import matplotlib.lines as mlines

from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles

from sklearn import svm

from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs
class InHouseSVM:

    # Inspired by https://habr.com/ru/company/ods/blog/484148/

    

    num_epochs = 200

    etha = 0.01

    alpha = 0.1

    

    def __init__(self, kernel):

        self.kernel = kernel



    def fit(self, X, Y):

        if self.kernel == 'poly':

            X = self._add_second_degree(X)

        X = np.hstack((X, np.ones((X.shape[0],1)))) # add bias

        Y = Y * 2 - 1 # convert [0, 1] to [-1, 1]

        w = np.random.normal(loc=0, scale=0.05, size=X.shape[1]) # random initial weights

        

        for epoch in range(self.num_epochs):

            for i, x in enumerate(X):

                margin = np.dot(w, x) * Y[i]

                if margin >= 1:

                    w -= self.etha * self.alpha*w / self.num_epochs

                else:

                    w -= self.etha * (self.alpha*w / self.num_epochs - Y[i]*x)

        self.w = w

        

    def decision_function(self, X):

        ans = np.zeros((X.shape[0]))

        for i,p in enumerate(X):

            x, y = p[0], p[1]

            w = self.w

            if self.kernel == 'linear':

                ans[i] = x*w[0] + y*w[1] + w[2]

            elif self.kernel == 'poly':

                ans[i] = x*w[0] + y*w[1] + x**2*w[2] + y**2*w[3] + x*y*w[4] + w[5]

        return ans

    

    def _add_second_degree(self, X):

        return np.hstack((X, X[:,[0]]**2, X[:,[1]]**2, X[:,[0]]*X[:,[1]]))
def get_toy_datsets():

    return [make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1),

            make_moons(noise=0.3, random_state=0),

            make_circles(noise=0.2, factor=0.5, random_state=1)]
def plot_svm_kernel(kernel):

    colormap = ListedColormap(['#0000FF', '#FF0000'])

    fig = plt.figure(figsize=(8, 12))

    

    for i, ds in enumerate(get_toy_datsets()):

        X, y = ds

        

        classifiers = ((InHouseSVM(kernel), 'In House'),

                       (svm.SVC(kernel=kernel), 'Out of the box'))

        

        for j, (cls, cls_name) in enumerate(classifiers):

            cls.fit(X, y)

            

            ax = plt.subplot(3, 2, i*2+j+1)

            ax.scatter(X[:, 0], X[:, 1], c = y, cmap=colormap)



            xs = np.arange(*ax.get_xbound(), 0.02)

            ys = np.arange(*ax.get_ybound(), 0.02)

            xx, yy = np.meshgrid(xs, ys)

            

            Z = cls.decision_function(np.c_[xx.ravel(), yy.ravel()])

            Z = Z.reshape(xx.shape)

            Z = np.sign(Z) # ignore absolute value

            

            ax.contourf(xs, ys, Z, alpha = 0.4, cmap = 'bwr')

            ax.text(xx.mean() , yy.max() + 0.1, cls_name,

                size=15, horizontalalignment='center')

            



plot_svm_kernel("linear")

plot_svm_kernel("poly")    