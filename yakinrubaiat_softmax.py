path = '../input/cs231n/'



import os

os.chdir(path)
import random

import numpy as np

from cs231n.data_utils import load_CIFAR10

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook



%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'



# for auto-reloading extenrnal modules

# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

%load_ext autoreload

%autoreload 2
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):

    """

    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare

    it for the linear classifier. These are the same steps as we used for the

    SVM, but condensed to a single function.  

    """

    # Load the raw CIFAR-10 data

    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'

    

    # Cleaning up variables to prevent loading data multiple times (which may cause memory issue)

    try:

       del X_train, y_train

       del X_test, y_test

       print('Clear previously loaded data.')

    except:

       pass



    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    

    # subsample the data

    mask = list(range(num_training, num_training + num_validation))

    X_val = X_train[mask]

    y_val = y_train[mask]

    mask = list(range(num_training))

    X_train = X_train[mask]

    y_train = y_train[mask]

    mask = list(range(num_test))

    X_test = X_test[mask]

    y_test = y_test[mask]

    mask = np.random.choice(num_training, num_dev, replace=False)

    X_dev = X_train[mask]

    y_dev = y_train[mask]

    

    # Preprocessing: reshape the image data into rows

    X_train = np.reshape(X_train, (X_train.shape[0], -1))

    X_val = np.reshape(X_val, (X_val.shape[0], -1))

    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    

    # Normalize the data: subtract the mean image

    mean_image = np.mean(X_train, axis = 0)

    X_train -= mean_image

    X_val -= mean_image

    X_test -= mean_image

    X_dev -= mean_image

    

    # add bias dimension and transform into columns

    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])

    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])

    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    

    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev





# Invoke the above function to get our data.

X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()

print('Train data shape: ', X_train.shape)

print('Train labels shape: ', y_train.shape)

print('Validation data shape: ', X_val.shape)

print('Validation labels shape: ', y_val.shape)

print('Test data shape: ', X_test.shape)

print('Test labels shape: ', y_test.shape)

print('dev data shape: ', X_dev.shape)

print('dev labels shape: ', y_dev.shape)
def softmax_loss_naive(W, X, y, reg):

    """

    Softmax loss function, naive implementation (with loops)



    Inputs have dimension D, there are C classes, and we operate on minibatches

    of N examples.



    Inputs:

    - W: A numpy array of shape (D, C) containing weights.

    - X: A numpy array of shape (N, D) containing a minibatch of data.

    - y: A numpy array of shape (N,) containing training labels; y[i] = c means

      that X[i] has label c, where 0 <= c < C.

    - reg: (float) regularization strength



    Returns a tuple of:

    - loss as single float

    - gradient with respect to weights W; an array of same shape as W

    """

    # Initialize the loss and gradient to zero.

    loss = 0.0

    dW = np.zeros_like(W)



    #############################################################################

    # TODO: Compute the softmax loss and its gradient using explicit loops.     #

    # Store the loss in loss and the gradient in dW. If you are not careful     #

    # here, it is easy to run into numeric instability. Don't forget the        #

    # regularization!                                                           #

    #############################################################################

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



    num_classes = W.shape[1]

    num_train = X.shape[0]

    

    for i in range(num_train):

        scores = X[i].dot(W)

        correct_class_score = scores[y[i]]

        

        max_score = scores.max()

        scores -= max_score

        

        loss += -correct_class_score + max_score + np.log(np.exp(scores).sum())

        

        for j in range(num_classes):

            dW[:,j] += np.exp(scores[j]) / np.exp(scores).sum() * X[i,:]

        

        dW[:,y[i]] -= X[i,:]

        

    loss /= num_train

    loss += reg * np.sum(W*W)

    

    dW /= num_train

    dW += reg * np.sum(W*W)



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



    return loss, dW
def softmax_loss_vectorized(W, X, y, reg):

    """

    Softmax loss function, vectorized version.



    Inputs and outputs are the same as softmax_loss_naive.

    """

    # Initialize the loss and gradient to zero.

    loss = 0.0

    dW = np.zeros_like(W)



    #############################################################################

    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #

    # Store the loss in loss and the gradient in dW. If you are not careful     #

    # here, it is easy to run into numeric instability. Don't forget the        #

    # regularization!                                                           #

    #############################################################################

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



    num_train = X.shape[0]

    scores = X.dot(W)

    

    correct_class_scores = scores[range(num_train),y]

    

    max_scores = scores.max(axis=1,keepdims=True)

    

    scores -= max_scores

    

    loss = -correct_class_scores.sum() + max_scores.sum() + np.log(np.exp(scores).sum(axis=1)).sum()

    loss /= num_train

    loss += reg * np.sum(W*W)

    

    

    softmax_deriv = (np.exp(scores)/np.exp(scores).sum(axis=1).reshape(-1,1))

    softmax_deriv[range(num_train),y] -= 1

    

    dW = X.T.dot(softmax_deriv)

    dW /= num_train

    dW += 2 * reg * W



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



    return loss, dW
# First implement the naive softmax loss function with nested loops.

# Open the file cs231n/classifiers/softmax.py and implement the

# softmax_loss_naive function.

import time



# Generate a random softmax weight matrix and use it to compute the loss.

W = np.random.randn(3073, 10) * 0.0001

loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)



# As a rough sanity check, our loss should be something close to -log(0.1).

print('loss: %f' % loss)

print('sanity check: %f' % (-np.log(0.1)))
# Complete the implementation of softmax_loss_naive and implement a (naive)

# version of the gradient that uses nested loops.

loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)



# As we did for the SVM, use numeric gradient checking as a debugging tool.

# The numeric gradient should be close to the analytic gradient.

from cs231n.gradient_check import grad_check_sparse

f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]

grad_numerical = grad_check_sparse(f, W, grad, 10)



# similar to SVM case, do another gradient check with regularization

loss, grad = softmax_loss_naive(W, X_dev, y_dev, 5e1)

f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 5e1)[0]

grad_numerical = grad_check_sparse(f, W, grad, 10)
# Now that we have a naive implementation of the softmax loss function and its gradient,

# implement a vectorized version in softmax_loss_vectorized.

# The two versions should compute the same results, but the vectorized version should be

# much faster.

tic = time.time()

loss_naive, grad_naive = softmax_loss_naive(W, X_dev, y_dev, 0.000005)

toc = time.time()

print('naive loss: %e computed in %fs' % (loss_naive, toc - tic))





tic = time.time()

loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, y_dev, 0.000005)

toc = time.time()

print('vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))



# As we did for the SVM, we use the Frobenius norm to compare the two versions

# of the gradient.

grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')

print('Loss difference: %f' % np.abs(loss_naive - loss_vectorized))

print('Gradient difference: %f' % grad_difference)
class LinearClassifier(object):



    def __init__(self):

        self.W = None



    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,

              batch_size=200, verbose=False):

        """

        Train this linear classifier using stochastic gradient descent.



        Inputs:

        - X: A numpy array of shape (N, D) containing training data; there are N

          training samples each of dimension D.

        - y: A numpy array of shape (N,) containing training labels; y[i] = c

          means that X[i] has label 0 <= c < C for C classes.

        - learning_rate: (float) learning rate for optimization.

        - reg: (float) regularization strength.

        - num_iters: (integer) number of steps to take when optimizing

        - batch_size: (integer) number of training examples to use at each step.

        - verbose: (boolean) If true, print progress during optimization.



        Outputs:

        A list containing the value of the loss function at each training iteration.

        """

        num_train, dim = X.shape

        num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes

        if self.W is None:

            # lazily initialize W

            self.W = 0.001 * np.random.randn(dim, num_classes)



        # Run stochastic gradient descent to optimize W

        loss_history = []

        for it in range(num_iters):

            

            idx = np.random.choice(range(X.shape[0]),size=batch_size)

            X_batch = X[idx,:]

            y_batch = y[idx]



            #########################################################################

            # TODO:                                                                 #

            # Sample batch_size elements from the training data and their           #

            # corresponding labels to use in this round of gradient descent.        #

            # Store the data in X_batch and their corresponding labels in           #

            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #

            # and y_batch should have shape (batch_size,)                           #

            #                                                                       #

            # Hint: Use np.random.choice to generate indices. Sampling with         #

            # replacement is faster than sampling without replacement.              #

            #########################################################################

            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****





            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



            # evaluate loss and gradient

            loss, grad = self.loss(X_batch, y_batch, reg)

            loss_history.append(loss)



            # perform parameter update

            #########################################################################

            # TODO:                                                                 #

            # Update the weights using the gradient and the learning rate.          #

            #########################################################################

            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



            self.W -= learning_rate * grad



            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



            if verbose and it % 100 == 0:

                print('iteration %d / %d: loss %f' % (it, num_iters, loss))



        return loss_history



    def predict(self, X):

        """

        Use the trained weights of this linear classifier to predict labels for

        data points.



        Inputs:

        - X: A numpy array of shape (N, D) containing training data; there are N

          training samples each of dimension D.



        Returns:

        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional

          array of length N, and each element is an integer giving the predicted

          class.

        """

        y_pred = np.zeros(X.shape[0])

        ###########################################################################

        # TODO:                                                                   #

        # Implement this method. Store the predicted labels in y_pred.            #

        ###########################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



        y_pred = np.argmax(X.dot(self.W),axis=1)



        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred



    def loss(self, X_batch, y_batch, reg):

        """

        Compute the loss function and its derivative.

        Subclasses will override this.



        Inputs:

        - X_batch: A numpy array of shape (N, D) containing a minibatch of N

          data points; each point has dimension D.

        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.

        - reg: (float) regularization strength.



        Returns: A tuple containing:

        - loss as a single float

        - gradient with respect to self.W; an array of the same shape as W

        """

        pass
class Softmax(LinearClassifier):

    """ A subclass that uses the Softmax + Cross-entropy loss function """



    def loss(self, X_batch, y_batch, reg):

        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
# Use the validation set to tune hyperparameters (regularization strength and

# learning rate). You should experiment with different ranges for the learning

# rates and regularization strengths; if you are careful you should be able to

# get a classification accuracy of over 0.35 on the validation set.



results = {}

best_val = -1

best_softmax = None

learning_rates = np.linspace(3e-7, 5e-7,3)

regularization_strengths = np.linspace(5e3, 5e4, 3)



################################################################################

# TODO:                                                                        #

# Use the validation set to set the learning rate and regularization strength. #

# This should be identical to the validation that you did for the SVM; save    #

# the best trained softmax classifer in best_softmax.                          #

################################################################################

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



for lr in tqdm_notebook(learning_rates):

    for reg in tqdm_notebook(regularization_strengths):

        softmax_clf = Softmax()

        _ = softmax_clf.train(X_train,y_train,learning_rate=lr,

                             reg=reg,

                             num_iters=1500,

                             verbose=False)

        y_train_pred = softmax_clf.predict(X_train)

        train_acc = np.mean(y_train==y_train_pred)

        y_val_pred = softmax_clf.predict(X_val)

        val_acc = np.mean(y_val==y_val_pred)

        results[(lr,reg)] = (train_acc,val_acc)

        

        if val_acc > best_val:

            best_val = val_acc

            best_softmax = softmax_clf



# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    

# Print out results.

for lr, reg in sorted(results):

    train_accuracy, val_accuracy = results[(lr, reg)]

    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (

                lr, reg, train_accuracy, val_accuracy))

    

print('best validation accuracy achieved during cross-validation: %f' % best_val)
# evaluate on test set

# Evaluate the best softmax on test set

y_test_pred = best_softmax.predict(X_test)

test_accuracy = np.mean(y_test == y_test_pred)

print('softmax on raw pixels final test set accuracy: %f' % (test_accuracy, ))
# Visualize the learned weights for each class

w = best_softmax.W[:-1,:] # strip out the bias

w = w.reshape(32, 32, 3, 10)



w_min, w_max = np.min(w), np.max(w)



classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for i in range(10):

    plt.subplot(2, 5, i + 1)

    

    # Rescale the weights to be between 0 and 255

    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)

    plt.imshow(wimg.astype('uint8'))

    plt.axis('off')

    plt.title(classes[i])