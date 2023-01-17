# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn import datasets

import matplotlib.pyplot as plt

import random

from sklearn.linear_model import Ridge, Lasso, ElasticNet

from sklearn.preprocessing import PolynomialFeatures



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
m = 1000

X = 6*np.random.rand(m, 1)-3

y = 0.5*X**2 - X**3/3 + 2 + np.random.randn(m, 1)

plt.figure(figsize=(7, 7))

plt.scatter(X, y)

plt.show()
lists = sorted(zip(*[X, y]))

X, y = list(zip(*lists))

degrees=[2, 4, 6, 8, 10]

polynomials = []

for i in degrees:

    PolyFeatures = PolynomialFeatures(degree=i)

    X_poly = PolyFeatures.fit_transform(X)

    polynomials.append(X_poly)



alphas=[0, 0.1, 0.5, 1, 2, 10]



fig, axs = plt.subplots(2,3, figsize=(20, 20), facecolor='w', edgecolor='k')

fig.subplots_adjust(hspace = .1, wspace=.001)

axs = axs.ravel()



def visualize(alpha):

    for i in range(len(alphas)):

        LassoReg = Lasso(alpha=i)

        LassoReg.fit(X_poly, y)

        y_poly = LassoReg.predict(X_poly)

        axs[i].scatter(X, y)

        axs[i].plot(X, y_poly, 'r', lw=4)

        axs[i].set_title('Alpha: '+str(alpha[i]))

    fig.show()

        

visualize(alphas)
fig, axis = plt.subplots(1,5, figsize=(20, 7))

fig.subplots_adjust(hspace = .5, wspace=.001)

axis.ravel()

for i in range(len(polynomials)):

    axis[i].scatter(X, y)

    LassoReg = Lasso(alpha=0.5)

    LassoReg.fit(polynomials[i], y)

    y_pred = LassoReg.predict(polynomials[i])

    axis[i].scatter(X, y)

    axis[i].plot(X, y_pred,'green', lw=3)

    axis[i].set_title('Degree: '+str(degrees[i]))
fig, axs = plt.subplots(2,3, figsize=(20, 20), facecolor='w', edgecolor='k')

fig.subplots_adjust(hspace = .1, wspace=.001)

axs = axs.ravel()



def visualize_2(alpha):

    for i in range(len(alphas)):

        RidgeReg = Ridge(alpha=i)

        RidgeReg.fit(X_poly, y)

        y_poly = RidgeReg.predict(X_poly)

        axs[i].scatter(X, y, color='pink')

        axs[i].plot(X, y_poly, 'r', lw=3)

        axs[i].set_title('Alpha: '+str(alpha[i]))

    fig.show()

        

visualize_2(alphas)
fig, axs = plt.subplots(2,3, figsize=(20, 20), facecolor='w', edgecolor='k')

fig.subplots_adjust(hspace = .1, wspace=.001)

axs = axs.ravel()



ratios = [0.1, 0.2, 0.5, 0.6, 0.8, 0.9]



def visualize_3(alpha):

    for i in range(len(ratios)):

        ElasticReg = ElasticNet(alpha=0.1, l1_ratio=i)

        ElasticReg.fit(X_poly, y)

        y_poly = ElasticReg.predict(X_poly)

        axs[i].scatter(X, y, color='green')

        axs[i].plot(X, y_poly, 'orange', lw=4)

        axs[i].set_title('Ratio: '+str(ratios[i]))

    fig.show()

        

visualize_3(alphas)