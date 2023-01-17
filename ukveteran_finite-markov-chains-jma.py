import quantecon as qe

import numpy as np

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

%matplotlib inline
ψ = (0.3, 0.7)           # probabilities over {0, 1}

cdf = np.cumsum(ψ)       # convert into cummulative distribution

qe.random.draw(cdf, 5)   # generate 5 independent draws from ψ
def mc_sample_path(P, ψ_0=None, sample_size=1_000):



    # set up

    P = np.asarray(P)

    X = np.empty(sample_size, dtype=int)



    # Convert each row of P into a cdf

    n = len(P)

    P_dist = [np.cumsum(P[i, :]) for i in range(n)]



    # draw initial state, defaulting to 0

    if ψ_0 is not None:

        X_0 = qe.random.draw(np.cumsum(ψ_0))

    else:

        X_0 = 0



    # simulate

    X[0] = X_0

    for t in range(sample_size - 1):

        X[t+1] = qe.random.draw(P_dist[X[t]])



    return X
P = [[0.4, 0.6],

     [0.2, 0.8]]
X = mc_sample_path(P, ψ_0=[0.1, 0.9], sample_size=100_000)

np.mean(X == 0)
P = [[0.9, 0.1, 0.0],

     [0.4, 0.4, 0.2],

     [0.1, 0.1, 0.8]]



mc = qe.MarkovChain(P, ('poor', 'middle', 'rich'))

mc.is_irreducible
P = [[1.0, 0.0, 0.0],

     [0.1, 0.8, 0.1],

     [0.0, 0.2, 0.8]]



mc = qe.MarkovChain(P, ('poor', 'middle', 'rich'))

mc.is_irreducible
P = [[0, 1, 0],

     [0, 0, 1],

     [1, 0, 0]]



mc = qe.MarkovChain(P)

mc.period
P = [[0.0, 1.0, 0.0, 0.0],

     [0.5, 0.0, 0.5, 0.0],

     [0.0, 0.5, 0.0, 0.5],

     [0.0, 0.0, 1.0, 0.0]]



mc = qe.MarkovChain(P)

mc.period
mc.is_aperiodic
P = [[0.4, 0.6],

     [0.2, 0.8]]



mc = qe.MarkovChain(P)

mc.stationary_distributions  # Show all stationary distributions
P = ((0.971, 0.029, 0.000),

     (0.145, 0.778, 0.077),

     (0.000, 0.508, 0.492))

P = np.array(P)



ψ = (0.0, 0.2, 0.8)        # Initial condition



fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(111, projection='3d')



ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1),

       xticks=(0.25, 0.5, 0.75),

       yticks=(0.25, 0.5, 0.75),

       zticks=(0.25, 0.5, 0.75))



x_vals, y_vals, z_vals = [], [], []

for t in range(20):

    x_vals.append(ψ[0])

    y_vals.append(ψ[1])

    z_vals.append(ψ[2])

    ψ = ψ @ P



ax.scatter(x_vals, y_vals, z_vals, c='r', s=60)

ax.view_init(30, 210)



mc = qe.MarkovChain(P)

ψ_star = mc.stationary_distributions[0]

ax.scatter(ψ_star[0], ψ_star[1], ψ_star[2], c='k', s=60)



plt.show()