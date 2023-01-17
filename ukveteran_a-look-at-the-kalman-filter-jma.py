import numpy as np

import matplotlib.cm as cm

import matplotlib.pyplot as plt

%matplotlib inline

from quantecon import Kalman, LinearStateSpace

from scipy.stats import norm

from scipy.integrate import quad

from numpy.random import multivariate_normal

from scipy.linalg import eigvals
# Set up the Gaussian prior density p

Σ = [[0.4, 0.3], [0.3, 0.45]]

Σ = np.matrix(Σ)

x_hat = np.matrix([0.2, -0.2]).T

# Define the matrices G and R from the equation y = G x + N(0, R)

G = [[1, 0], [0, 1]]

G = np.matrix(G)

R = 0.5 * Σ

# The matrices A and Q

A = [[1.2, 0], [0, -0.2]]

A = np.matrix(A)

Q = 0.3 * Σ

# The observed value of y

y = np.matrix([2.3, -1.9]).T



# Set up grid for plotting

x_grid = np.linspace(-1.5, 2.9, 100)

y_grid = np.linspace(-3.1, 1.7, 100)

X, Y = np.meshgrid(x_grid, y_grid)



def bivariate_normal(x, y, σ_x=1.0, σ_y=1.0, μ_x=0.0, μ_y=0.0, σ_xy=0.0):



    x_μ = x - μ_x

    y_μ = y - μ_y



    ρ = σ_xy / (σ_x * σ_y)

    z = x_μ**2 / σ_x**2 + y_μ**2 / σ_y**2 - 2 * ρ * x_μ * y_μ / (σ_x * σ_y)

    denom = 2 * np.pi * σ_x * σ_y * np.sqrt(1 - ρ**2)

    return np.exp(-z / (2 * (1 - ρ**2))) / denom



def gen_gaussian_plot_vals(μ, C):

    "Z values for plotting the bivariate Gaussian N(μ, C)"

    m_x, m_y = float(μ[0]), float(μ[1])

    s_x, s_y = np.sqrt(C[0, 0]), np.sqrt(C[1, 1])

    s_xy = C[0, 1]

    return bivariate_normal(X, Y, s_x, s_y, m_x, m_y, s_xy)



# Plot the figure



fig, ax = plt.subplots(figsize=(10, 8))

ax.grid()



Z = gen_gaussian_plot_vals(x_hat, Σ)

ax.contourf(X, Y, Z, 6, alpha=0.6, cmap=cm.jet)

cs = ax.contour(X, Y, Z, 6, colors="black")

ax.clabel(cs, inline=1, fontsize=10)



plt.show()
fig, ax = plt.subplots(figsize=(10, 8))

ax.grid()



Z = gen_gaussian_plot_vals(x_hat, Σ)

ax.contourf(X, Y, Z, 6, alpha=0.6, cmap=cm.jet)

cs = ax.contour(X, Y, Z, 6, colors="black")

ax.clabel(cs, inline=1, fontsize=10)

ax.text(float(y[0]), float(y[1]), "$y$", fontsize=20, color="black")



plt.show()