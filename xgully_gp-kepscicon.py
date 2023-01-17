#!/usr/bin/env python

import numpy as np

import time

from scipy.linalg import cho_factor, cho_solve

import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'retina'
np.random.seed(123456)
true_m, true_b = 0.5, -0.25

N = 500

x = np.linspace(-5, 5, N)

y = true_m * x + true_b
yerr = 0.1 + 0.4 * np.random.rand(N)

yerr_hom = 0.4*np.ones(N)

hom_cov = np.diag(yerr_hom ** 2)

iid_cov = np.diag(yerr ** 2)

true_amp = 0.1

true_ell = 0.9

true_cov = true_amp * np.exp(-0.5 * (x[:, None]-x[None, :])**2 / true_ell**2) + np.diag(yerr ** 2)

y_noised = np.random.multivariate_normal(y, true_cov)
plt.plot(x, y_noised, label = 'Noisy signal')

plt.plot(x, y, color = 'k', label = 'True signal')

plt.legend();
plt.plot(x, y_noised-y, 'k.',label = 'Residual')

plt.legend();
A = np.vander(x, 2)

AT= A.T

C = iid_cov



t0 = time.time()

factor = cho_factor(C, overwrite_a=True)

S_inv = np.dot(AT, cho_solve(factor, A))

S = np.linalg.inv(S_inv)

ls_m, ls_b = np.dot(S, np.dot(AT, cho_solve(factor, y_noised)))

t1 = time.time()



net_time = t1-t0

print(" m: {:.2f} \n b: {:.2f} \n time: {}".format(ls_m, ls_b, net_time))
y_fit = y = ls_m * x + ls_b
plt.plot(x, y_noised, label = 'Noisy signal')

plt.plot(x, y_fit, color = 'r', label = 'Fitted signal')

plt.legend();