# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

from matplotlib import pyplot as plt

import warnings

warnings.filterwarnings("ignore")
def plot_graph(func, derivative):

    """Function for plotting activation and its derivative"""

    fig = plt.figure(figsize=(20,5))

    ax1 = fig.add_subplot(1,2,1)

    ax1.set_title('Activation function')

    plt.plot(func)

    ax2 = fig.add_subplot(1,2,2)

    ax2.set_title('Derivative')

    plt.plot(derivative)

    plt.show()
def sigmoid(scores):

    """sigmoid function"""

    return (1/(1 + np.exp(-scores)))



def sigmoid_grad(scores):

    """sigmoid function derivative"""

    return sigmoid(scores)*(1-sigmoid(scores))
"""Plot sigmoid and its derivative"""

scores = np.linspace(-10,10,100)

plot_graph(sigmoid(scores), sigmoid_grad(scores))
def tanh(scores):

    """tanh function"""

    exp_val = np.exp(-2*scores)

    return (2/(1 + exp_val))-1



def tanh1(scores):

    """alternate implementation of tanh function"""

    return (np.exp(scores) - np.exp(-scores))/(np.exp(scores) + np.exp(-scores))



def tanh_grad(scores):

    """tanh derivative function"""

    return 1 - np.power(tanh(scores), 2)
"""Plot tanh and its derivative"""

scores = np.linspace(-10,10,100)

plot_graph(tanh(scores), tanh_grad(scores))
def relu(scores):

    """Return 0 if scores < 0 otherwise keep scores as it is."""

    scores[scores<=0]=0

    return scores



def relu_grad(scores):

    """Return 1 if scores > 0 otherwise return 0"""

    scores = np.where(scores>0, 1 , 0)

    return scores
"""Plot ReLU function and its derivative"""

scores = np.linspace(-10,10,100)

plot_graph(relu(scores), relu_grad(scores))
def leaky_relu(scores, alpha):

    """Return scores if >= 0 otherwise return scores*alpha for scores < 0"""

    scores = np.where(scores>0,scores,scores * alpha)

    return scores



def leaky_relu_grad(scores, alpha):

    """Return 1 if scores > 0 else return alpha"""

    scores = np.where(scores > 0, 1, alpha)

    return scores
"""Plot leakyReLU and its derivative"""

scores = np.linspace(-20,10,100)

alpha = 0.01

plot_graph(leaky_relu(scores, alpha), leaky_relu_grad(scores, alpha))
# Softmax 

def softmax(scores):

    """Softmax function"""

    exp_scores = np.exp(scores)

    return exp_scores/np.sum(exp_scores)





# Source https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

def stablesoftmax(scores):

    """Compute the softmax of vector scores in a numerically stable way."""

    shift_scores = scores - np.max(scores)

    exps = np.exp(shift_scores)

    return exps / np.sum(exps)
scores = [0.1,2.5,0.3,4.2]

grad = softmax(scores)

print(f'{grad}')
# Lets run same function with some bigger values. As mentioned above output is nan.

scores_big_values = [1000,1500,1200]

grad_big_values = softmax(scores_big_values)

print(f'{grad_big_values}')

# Run same scores with stablesoftmax()

print(f'{stablesoftmax(scores_big_values)}')
# Derivative of softmax

# Source https://medium.com/@aerinykim/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d



def softmax_grad(softmax):

    """Derivative of softmax function.

    Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication

    """

    s = softmax.reshape(-1,1)

    return np.diagflat(s) - np.dot(s, s.T)



# It will return matrix of same shape as shape of matrix on L-1 layer i.e just before softmax layer/output layer.

softmax_grad(softmax(scores))