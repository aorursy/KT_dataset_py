%matplotlib inline

import theano

floatX = theano.config.floatX

import pymc3 as pm

import theano.tensor as T

import sklearn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from warnings import filterwarnings

sns.set_style('white')

from sklearn import datasets

from sklearn.preprocessing import scale

from sklearn.datasets import make_moons
X, Y = make_moons(noise=0.0, random_state=0, n_samples=100)

X = scale(X)
fig, ax = plt.subplots()

ax.scatter(X[Y==0, 0], X[Y==0, 1], label='Class 0')

ax.scatter(X[Y==1, 0], X[Y==1, 1], color='r', label='Class 1')

sns.despine(); ax.legend()

ax.set(xlabel='$X_1$', ylabel='$X_2$', title='Toy binary classification data set');
def construct_nn(ann_input, ann_output):

    n_hidden = 5



    # Initialize random weights between each layer

    init_1 = np.random.randn(X.shape[1], n_hidden).astype(floatX)

    init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)

    init_out = np.random.randn(n_hidden).astype(floatX)



    with pm.Model() as neural_network:

        # Trick: Turn inputs and outputs into shared variables using the data container pm.Data

        # It's still the same thing, but we can later change the values of the shared variable

        # (to switch in the test-data later) and pymc3 will just use the new data.

        # Kind-of like a pointer we can redirect.

        # For more info, see: http://deeplearning.net/software/theano/library/compile/shared.html

        ann_input = pm.Data('ann_input', X)

        ann_output = pm.Data('ann_output', Y)



        # Weights from input to hidden layer

        weights_in_1 = pm.Normal('w_in_1', 0, sigma=1,

                                 shape=(X.shape[1], n_hidden),

                                 testval=init_1)



        # Weights from 1st to 2nd layer

        weights_1_2 = pm.Normal('w_1_2', 0, sigma=1,

                                shape=(n_hidden, n_hidden),

                                testval=init_2)



        # Weights from hidden layer to output

        weights_2_out = pm.Normal('w_2_out', 0, sigma=1,

                                  shape=(n_hidden,),

                                  testval=init_out)



        # Build neural-network using tanh activation function

        act_1 = pm.math.tanh(pm.math.dot(ann_input,

                                         weights_in_1))

        act_2 = pm.math.tanh(pm.math.dot(act_1,

                                         weights_1_2))

        act_out = pm.math.sigmoid(pm.math.dot(act_2,

                                              weights_2_out))



        # Binary classification -> Bernoulli likelihood

        out = pm.Bernoulli('out',

                           act_out,

                           observed=ann_output,

                           total_size=Y.shape[0] # IMPORTANT for minibatches

                          )

    return neural_network



eural_network = construct_nn(X, Y)
from pymc3.theanof import set_tt_rng, MRG_RandomStreams

set_tt_rng(MRG_RandomStreams(42))
%%time

with neural_network:

    inference = pm.ADVI()

    approx = pm.fit(n=30000, method=inference)
plt.plot(inference.hist, label='new ADVI', alpha=.6)

plt.legend()

plt.ylabel('ELBO')

plt.xlabel('iteration')
trace = approx.sample(draws=5000)
# create symbolic input for lazy evaluation

x = T.matrix('X')



# symbolic number of samples is supported

n = T.iscalar('n')



# Do not forget test_value or set theano.config.compute_test_value = 'off'

x.tag.test_value = np.empty_like(X[:10])

n.tag.test_value = 100

_sample_proba = approx.sample_node(neural_network.out.distribution.p,

                                   size=n,

                                   more_replacements={neural_network['ann_input']: x})

# It is time to compile the function

# No updates are needed for Approximation random generator

# Efficient vectorized form of sampling is used

sample_proba = theano.function([x, n], _sample_proba)
sample_proba(X, 500)
pred = sample_proba(X, 500).mean(0) > 0.5
pred.shape
print('Accuracy = {}%'.format((Y == pred).mean() * 100))
fig, ax = plt.subplots()

ax.scatter(X[pred==True, 0], X[pred==True, 1])

ax.scatter(X[pred==False, 0], X[pred==False, 1], color='r')

sns.despine()

ax.set(title='Predicted labels in testing set', xlabel='X', ylabel='Y');
grid = pm.floatX(np.mgrid[-3:3:100j,-3:3:100j])

grid_2d = grid.reshape(2, -1).T

dummy_out = np.ones(grid.shape[1], dtype=np.int8)
ppc = sample_proba(grid_2d ,500)
cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)

fig, ax = plt.subplots(figsize=(16, 9))

contour = ax.contourf(grid[0], grid[1], ppc.mean(axis=0).reshape(100, 100), cmap=cmap)

ax.scatter(X[pred==0, 0], X[pred==0, 1])

ax.scatter(X[pred==1, 0], X[pred==1, 1], color='r')

cbar = plt.colorbar(contour, ax=ax)

_ = ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel='X', ylabel='Y');

cbar.ax.set_ylabel('Posterior predictive mean probability of class label = 0');

cmap = sns.cubehelix_palette(light=1, as_cmap=True)

fig, ax = plt.subplots(figsize=(16, 9))

contour = ax.contourf(grid[0], grid[1], ppc.std(axis=0).reshape(100, 100), cmap=cmap)

ax.scatter(X[pred==0, 0], X[pred==0, 1])

ax.scatter(X[pred==1, 0], X[pred==1, 1], color='r')

cbar = plt.colorbar(contour, ax=ax)

_ = ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel='X', ylabel='Y');

cbar.ax.set_ylabel('Uncertainty (posterior predictive standard deviation)');