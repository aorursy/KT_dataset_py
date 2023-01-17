import numpy as np

import scipy.stats

import matplotlib.pyplot as plt

import pandas as pd

import pymc3 as pm

import theano.tensor as tt



import warnings

warnings.filterwarnings("ignore")



%matplotlib inline
xs = np.linspace(-8., 12., 100)

plt.plot(xs, \

    .2 * scipy.stats.norm.pdf(xs, loc=-3., scale=np.sqrt(2.)) + \

    .8 * scipy.stats.norm.pdf(xs, loc=3., scale=np.sqrt(5.)));
particles = []

for i in range(5000):

    if scipy.stats.uniform.rvs() <= .2:

        particles.append(scipy.stats.norm.rvs(loc=-3., scale=np.sqrt(2.)))

    else:

        particles.append(scipy.stats.norm.rvs(loc=3., scale=np.sqrt(5.)))
xs = np.linspace(-8., 12., 100)

plt.plot(xs, \

    .2 * scipy.stats.norm.pdf(xs, loc=-3., scale=np.sqrt(2.)) + \

    .8 * scipy.stats.norm.pdf(xs, loc=3., scale=np.sqrt(5.)))

plt.hist(particles, density=True, bins=100);

plt.plot(particles, [0 for p in particles], '+', markersize=10);