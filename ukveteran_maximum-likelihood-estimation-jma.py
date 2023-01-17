import numpy as np

from numpy import exp

import matplotlib.pyplot as plt

%matplotlib inline

from scipy.special import factorial

import pandas as pd

from mpl_toolkits.mplot3d import Axes3D

import statsmodels.api as sm

from statsmodels.api import Poisson

from scipy import stats

from scipy.stats import norm

from statsmodels.iolib.summary2 import summary_col
poisson_pmf = lambda y, μ: μ**y / factorial(y) * exp(-μ)

y_values = range(0, 25)



fig, ax = plt.subplots(figsize=(12, 8))



for μ in [1, 5, 10]:

    distribution = []

    for y_i in y_values:

        distribution.append(poisson_pmf(y_i, μ))

    ax.plot(y_values,

            distribution,

            label=f'$\mu$={μ}',

            alpha=0.5,

            marker='o',

            markersize=8)



ax.grid()

ax.set_xlabel('$y$', fontsize=14)

ax.set_ylabel('$f(y \mid \mu)$', fontsize=14)

ax.axis(xmin=0, ymin=0)

ax.legend(fontsize=14)



plt.show()
y_values = range(0, 20)



# Define a parameter vector with estimates

β = np.array([0.26, 0.18, 0.25, -0.1, -0.22])



# Create some observations X

datasets = [np.array([0, 1, 1, 1, 2]),

            np.array([2, 3, 2, 4, 0]),

            np.array([3, 4, 5, 3, 2]),

            np.array([6, 5, 4, 4, 7])]





fig, ax = plt.subplots(figsize=(12, 8))



for X in datasets:

    μ = exp(X @ β)

    distribution = []

    for y_i in y_values:

        distribution.append(poisson_pmf(y_i, μ))

    ax.plot(y_values,

            distribution,

            label=f'$\mu_i$={μ:.1}',

            marker='o',

            markersize=8,

            alpha=0.5)



ax.grid()

ax.legend()

ax.set_xlabel('$y \mid x_i$')

ax.set_ylabel(r'$f(y \mid x_i; \beta )$')

ax.axis(xmin=0, ymin=0)

plt.show()
def plot_joint_poisson(μ=7, y_n=20):

    yi_values = np.arange(0, y_n, 1)



    # Create coordinate points of X and Y

    X, Y = np.meshgrid(yi_values, yi_values)



    # Multiply distributions together

    Z = poisson_pmf(X, μ) * poisson_pmf(Y, μ)



    fig = plt.figure(figsize=(12, 8))

    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z.T, cmap='terrain', alpha=0.6)

    ax.scatter(X, Y, Z.T, color='black', alpha=0.5, linewidths=1)

    ax.set(xlabel='$y_1$', ylabel='$y_2$')

    ax.set_zlabel('$f(y_1, y_2)$', labelpad=10)

    plt.show()



plot_joint_poisson(μ=7, y_n=20)
X = np.array([[1, 2, 5],

              [1, 1, 3],

              [1, 4, 2],

              [1, 5, 2],

              [1, 3, 1]])



y = np.array([1, 0, 1, 1, 0])



stats_poisson = Poisson(y, X).fit()

print(stats_poisson.summary())