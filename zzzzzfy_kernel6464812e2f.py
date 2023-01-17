# As usual, a bit of setup

from __future__ import print_function

import time

import numpy as np

import matplotlib.pyplot as plt

from MS326.classifiers.fc_net import *

from MS326.data_utils import get_mnist_data

from MS326.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array

from MS326.solver import Solver



%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'



# for auto-reloading external modules

# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

%load_ext autoreload

%autoreload 2



def rel_error(x, y):

  """ returns relative error """

  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
# Load the (preprocessed) CIFAR10 data.



data = get_mnist_data()

for k, v in data.items():

  print('%s: ' % k, v.shape)
def dropout_forward(x, dropout_param):

    """

    Performs the forward pass for (inverted) dropout.

    Inputs:

    - x: Input data, of any shape

    - dropout_param: A dictionary with the following keys:

      - p: Dropout parameter. We keep each neuron output with probability p.

      - mode: 'test' or 'train'. If the mode is train, then perform dropout;

        if the mode is test, then just return the input.

      - seed: Seed for the random number generator. Passing seed makes this

        function deterministic, which is needed for gradient checking but not

        in real networks.

    Outputs:

    - out: Array of the same shape as x.

    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout

      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.

    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron

    output; this might be contrary to some sources, where it is referred to

    as the probability of dropping a neuron output.

    """

    p, mode = dropout_param['p'], dropout_param['mode']

    if 'seed' in dropout_param:

        np.random.seed(dropout_param['seed'])



    mask = None



    if mode == 'train':

        mask = (np.random.rand(*x.shape) < p) / p # <p keep prob

        out = x * mask

    elif mode == 'test':

        out = x



    cache = (dropout_param, mask)

    out = out.astype(x.dtype, copy=False)



    return out, cache
np.random.seed(326)

x = np.random.randn(500, 500) + 10



for p in [0.25, 0.4, 0.7]: 

  out, _ = dropout_forward(x, {'mode': 'train', 'p': p})

  out_test, _ = dropout_forward(x, {'mode': 'test', 'p': p})



  print('Running tests with p = ', p)

  print('Mean of input: ', x.mean())

  print('Mean of train-time output: ', out.mean())

  print('Mean of test-time output: ', out_test.mean())

  print('Fraction of train-time output set to zero: ', (out == 0).mean())

  print('Fraction of test-time output set to zero: ', (out_test == 0).mean())

  print()
def dropout_backward(dout, cache):

    """

    Perform the backward pass for (inverted) dropout.

    Inputs:

    - dout: Upstream derivatives, of any shape

    - cache: (dropout_param, mask) from dropout_forward.

    """

    dropout_param, mask = cache

    mode = dropout_param['mode']



    if mode == 'train':

        dx = dout * mask 

    elif mode == 'test':

        dx = dout

    return dx
np.random.seed(326)

x = np.random.randn(10, 10) + 10

dout = np.random.randn(*x.shape)



dropout_param = {'mode': 'train', 'p': 0.2, 'seed': 123}

out, cache = dropout_forward(x, dropout_param)

dx = dropout_backward(dout, cache)

dx_num = eval_numerical_gradient_array(lambda xx: dropout_forward(xx, dropout_param)[0], x, dout)



# Error should be around e-10 or less

print('dx relative error: ', rel_error(dx, dx_num))
np.random.seed(326)

N, D, H1, H2, C = 2, 15, 20, 30, 10

X = np.random.randn(N, D)

y = np.random.randint(C, size=(N,))



for dropout in [1, 0.75, 0.5]:

  print('Running check with dropout = ', dropout)

  model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,

                            weight_scale=5e-2, dtype=np.float64,

                            dropout=dropout, seed=123)



  loss, grads = model.loss(X, y)

  print('Initial loss: ', loss)

  

  # Relative errors should be around e-6 or less; Note that it's fine

  # if for dropout=1 you have W2 error be on the order of e-5.

  for name in sorted(grads):

    f = lambda _: model.loss(X, y)[0]

    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)

    print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))

  print()
# Train two identical nets, one with dropout and one without

np.random.seed(326)

num_train = 500

small_data = {

  'X_train': data['X_train'][:num_train],

  'y_train': data['y_train'][:num_train],

  'X_val': data['X_val'],

  'y_val': data['y_val'],

}



solvers = {}

dropout_choices = [1, 0.25]

for dropout in dropout_choices:

  model = FullyConnectedNet([500], dropout=dropout)

  print(dropout)



  solver = Solver(model, small_data,

                  num_epochs=25, batch_size=100,

                  update_rule='adam',

                  optim_config={

                    'learning_rate': 5e-4,

                  },

                  verbose=True, print_every=100)

  solver.train()

  solvers[dropout] = solver

  print()
# Plot train and validation accuracies of the two models



train_accs = []

val_accs = []

for dropout in dropout_choices:

  solver = solvers[dropout]

  train_accs.append(solver.train_acc_history[-1])

  val_accs.append(solver.val_acc_history[-1])



plt.subplot(3, 1, 1)

for dropout in dropout_choices:

  plt.plot(solvers[dropout].train_acc_history, 'o', label='%.2f dropout' % dropout)

plt.title('Train accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(ncol=2, loc='lower right')

  

plt.subplot(3, 1, 2)

for dropout in dropout_choices:

  plt.plot(solvers[dropout].val_acc_history, 'o', label='%.2f dropout' % dropout)

plt.title('Val accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(ncol=2, loc='lower right')



plt.gcf().set_size_inches(15, 15)

plt.show()