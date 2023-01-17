import pandas as pd

import numpy as np

data = pd.read_csv('/kaggle/input/sparse-recommender-system-data-movie-ratings/converted.csv')

data.head()
data = data[data.columns[:1000]].loc[range(1,1001)]

data.head()
import numpy

import numpy as np

import pandas as pd

import progressbar as pb



def matrix_factorization(R, P, Q, K, steps=1000, alpha=0.0002, beta=0.02):

    Q = Q.T

    for step in pb.progressbar(range(steps)):

        for i in range(len(R)):

            for j in range(len(R[i])):

                if R[i][j] > 0:

                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])

                    for k in range(K):

                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])

                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        eR = numpy.dot(P,Q)

        e = 0

        for i in range(len(R)):

            for j in range(len(R[i])):

                if R[i][j] > 0:

                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)

                    for k in range(K):

                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))

        if e < 0.001:

            break

    return P, Q.T
R = np.array(data)

N = len(R)

M = len(R[0])

K = 2



P = np.random.rand(N,K)

Q = np.random.rand(M,K)



nP, nQ = matrix_factorization(R, P, Q, K)

nR = numpy.dot(nP, nQ.T)
pd.DataFrame(nR).to_csv('predictions.csv')
nR