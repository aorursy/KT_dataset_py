import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import scipy.linalg as la



from quantecon import Kalman

from quantecon import LinearStateSpace

from scipy.stats import norm

np.set_printoptions(linewidth=120, precision=4, suppress=True)
# Make some parameter choices

# sigx/sigy are state noise std err and measurement noise std err

μ_0, σ_x, σ_y = 10, 1, 5



# Create a LinearStateSpace object

A, C, G, H = 1, σ_x, 1, σ_y

ss = LinearStateSpace(A, C, G, H, mu_0=μ_0)



# Set prior and initialize the Kalman type

x_hat_0, Σ_0 = 10, 1

kmuth = Kalman(ss, x_hat_0, Σ_0)



# Computes stationary values which we need for the innovation

# representation

S1, K1 = kmuth.stationary_values()



# Form innovation representation state-space

Ak, Ck, Gk, Hk = A, K1, G, 1



ssk = LinearStateSpace(Ak, Ck, Gk, Hk, mu_0=x_hat_0)
# Create grand state-space for y_t, a_t as observed vars -- Use

# stacking trick above

Af = np.array([[ 1,      0,        0],

               [K1, 1 - K1, K1 * σ_y],

               [ 0,      0,        0]])

Cf = np.array([[σ_x,        0],

               [  0, K1 * σ_y],

               [  0,        1]])

Gf = np.array([[1,  0, σ_y],

               [1, -1, σ_y]])



μ_true, μ_prior = 10, 10

μ_f = np.array([μ_true, μ_prior, 0]).reshape(3, 1)



# Create the state-space

ssf = LinearStateSpace(Af, Cf, Gf, mu_0=μ_f)



# Draw observations of y from the state-space model

N = 50

xf, yf = ssf.simulate(N)



print(f"Kalman gain = {K1}")

print(f"Conditional variance = {S1}")
fig, ax = plt.subplots()

ax.plot(xf[0, :], label="$x_t$")

ax.plot(xf[1, :], label="Filtered $x_t$")

ax.legend()

ax.set_xlabel("Time")

ax.set_title(r"$x$ vs $\hat{x}$")

plt.show()
fig, ax = plt.subplots()

ax.plot(yf[0, :], label="y")

ax.plot(xf[0, :], label="x")

ax.legend()

ax.set_title(r"$x$ and $y$")

ax.set_xlabel("Time")

plt.show()
# Kalman Methods for MA and VAR

coefs_ma = kmuth.stationary_coefficients(5, "ma")

coefs_var = kmuth.stationary_coefficients(5, "var")



# Coefficients come in a list of arrays, but we

# want to plot them and so need to stack into an array

coefs_ma_array = np.vstack(coefs_ma)

coefs_var_array = np.vstack(coefs_var)



fig, ax = plt.subplots(2)

ax[0].plot(coefs_ma_array, label="MA")

ax[0].legend()

ax[1].plot(coefs_var_array, label="VAR")

ax[1].legend()



plt.show()