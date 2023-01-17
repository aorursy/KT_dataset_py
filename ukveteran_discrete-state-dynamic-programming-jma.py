import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import quantecon as qe

import scipy.sparse as sparse

from quantecon import compute_fixed_point

from quantecon.markov import DiscreteDP
class SimpleOG:



    def __init__(self, B=10, M=5, α=0.5, β=0.9):

        """

        Set up R, Q and β, the three elements that define an instance of

        the DiscreteDP class.

        """



        self.B, self.M, self.α, self.β  = B, M, α, β

        self.n = B + M + 1

        self.m = M + 1



        self.R = np.empty((self.n, self.m))

        self.Q = np.zeros((self.n, self.m, self.n))



        self.populate_Q()

        self.populate_R()



    def u(self, c):

        return c**self.α



    def populate_R(self):

        """

        Populate the R matrix, with R[s, a] = -np.inf for infeasible

        state-action pairs.

        """

        for s in range(self.n):

            for a in range(self.m):

                self.R[s, a] = self.u(s - a) if a <= s else -np.inf



    def populate_Q(self):

        """

        Populate the Q matrix by setting



            Q[s, a, s'] = 1 / (1 + B) if a <= s' <= a + B



        and zero otherwise.

        """



        for a in range(self.m):

            self.Q[:, a, a:(a + self.B + 1)] = 1.0 / (self.B + 1)
g = SimpleOG()  # Use default parameters
ddp = qe.markov.DiscreteDP(g.R, g.Q, g.β)
results = ddp.solve(method='policy_iteration')
dir(results)
results.v
results.sigma
results.max_iter
results.num_iter
results.mc.stationary_distributions
ddp = qe.markov.DiscreteDP(g.R, g.Q, 0.99)  # Increase β to 0.99

results = ddp.solve(method='policy_iteration')

results.mc.stationary_distributions