import numpy as np

import quantecon as qe

import matplotlib.pyplot as plt



%matplotlib inline
α = 4.0
def qm(x0, n):

    x = np.empty(n+1)

    x[0] = x0

    for t in range(n):

      x[t+1] = α * x[t] * (1 - x[t])

    return x



x = qm(0.1, 250)

fig, ax = plt.subplots()

ax.plot(x, 'b-', lw=2, alpha=0.8)

ax.set_xlabel('$t$', fontsize=12)

ax.set_ylabel('$x_{t}$', fontsize = 12)

plt.show()
from numba import jit



qm_numba = jit(qm)
n = 10_000_000

qe.tic()

qm(0.1, int(n))

time1 = qe.toc()
qm_numba(0.1, int(n))

time2 = qe.toc()
qe.tic()

qm_numba(0.1, int(n))

time3 = qe.toc()
time1 / time3  # Calculate speed gain