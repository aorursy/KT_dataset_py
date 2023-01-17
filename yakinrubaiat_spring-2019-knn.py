path = '../input/cs231n/'

import os

os.chdir(path)
# Run some setup code for this notebook.



import random

import numpy as np

from cs231n.data_utils import load_CIFAR10

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook



# This is a bit of magic to make matplotlib figures appear inline in the notebook

# rather than in a new window.

%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'



# Some more magic so that the notebook will reload external python modules;

# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

%load_ext autoreload

%autoreload 2
# Load the raw CIFAR-10 data.

cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'



# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)

try:

   del X_train, y_train

   del X_test, y_test

   print('Clear previously loaded data.')

except:

   pass



X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)



# As a sanity check, we print out the size of the training and test data.

print('Training data shape: ', X_train.shape)

print('Training labels shape: ', y_train.shape)

print('Test data shape: ', X_test.shape)

print('Test labels shape: ', y_test.shape)
# Visualize some examples from the dataset.

# We show a few examples of training images from each class.

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

num_classes = len(classes)

samples_per_class = 10

for y, cls in enumerate(classes):

    idxs = np.flatnonzero(y_train == y)

    idxs = np.random.choice(idxs, samples_per_class, replace=False)

    for i, idx in enumerate(idxs):

        plt_idx = i * num_classes + y + 1

        plt.subplot(samples_per_class, num_classes, plt_idx)

        plt.imshow(X_train[idx].astype('uint8'))

        plt.axis('off')

        if i == 0:

            plt.title(cls)

plt.show()
# Subsample the data for more efficient code execution in this exercise

num_training = 5000

mask = list(range(num_training))

X_train = X_train[mask]

y_train = y_train[mask]



num_test = 500

mask = list(range(num_test))

X_test = X_test[mask]

y_test = y_test[mask]



print(X_train.shape[0])

# Reshape the image data into rows

X_train = np.reshape(X_train, (X_train.shape[0], -1))

X_test = np.reshape(X_test, (X_test.shape[0], -1))

print(X_train.shape, X_test.shape)
X_norm = np.sum(X_test ** 2, axis = 1).reshape(-1,1)



X_train_norm = np.sum(X_train ** 2, axis = 1)

X_train_norm.shape
sum1 = X_norm + X_train_norm

sum1.shape
import numpy as np



class KNearestNeighbor(object):

  """ a kNN classifier with L2 distance """



  def __init__(self):

    self.x_train = None

    self.y_train = None



  def train(self, X, y):

    """

    Train the classifier. For k-nearest neighbors this is just 

    memorizing the training data.



    Inputs:

    - X: A numpy array of shape (num_train, D) containing the training data

      consisting of num_train samples each of dimension D.

    - y: A numpy array of shape (N,) containing the training labels, where

         y[i] is the label for X[i].

    """

    self.X_train = X

    self.y_train = y

    

  def predict(self, X, k=1, num_loops=0):

    """

    Predict labels for test data using this classifier.



    Inputs:

    - X: A numpy array of shape (num_test, D) containing test data consisting

         of num_test samples each of dimension D.

    - k: The number of nearest neighbors that vote for the predicted labels.

    - num_loops: Determines which implementation to use to compute distances

      between training points and testing points.



    Returns:

    - y: A numpy array of shape (num_test,) containing predicted labels for the

      test data, where y[i] is the predicted label for the test point X[i].  

    """

    if num_loops == 0:

      dists = self.compute_distances_no_loops(X)

    elif num_loops == 1:

      dists = self.compute_distances_one_loop(X)

    elif num_loops == 2:

      dists = self.compute_distances_two_loops(X)

    else:

      raise ValueError('Invalid value %d for num_loops' % num_loops)



    return self.predict_labels(dists, k=k)



  def compute_distances_two_loops(self, X):

    """

    Compute the distance between each test point in X and each training point

    in self.X_train using a nested loop over both the training data and the 

    test data.



    Inputs:

    - X: A numpy array of shape (num_test, D) containing test data.



    Returns:

    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]

      is the Euclidean distance between the ith test point and the jth training

      point.

    """

    num_test = X.shape[0]

    num_train = self.X_train.shape[0]

    dists = np.zeros((num_test, num_train))

    for i in range(num_test):

      for j in range(num_train):

        #####################################################################

        # TODO:                                                             #

        # Compute the l2 distance between the ith test point and the jth    #

        # training point, and store the result in dists[i, j]. You should   #

        # not use a loop over dimension.                                    #

        #####################################################################

        dists[i,j] = np.linalg.norm(X[i,:] - self.X_train[j,:])

        #####################################################################

        #                       END OF YOUR CODE                            #

        #####################################################################

    return dists



  def compute_distances_one_loop(self, X):

    """

    Compute the distance between each test point in X and each training point

    in self.X_train using a single loop over the test data.



    Input / Output: Same as compute_distances_two_loops

    """

    num_test = X.shape[0]

    num_train = self.X_train.shape[0]

    dists = np.zeros((num_test, num_train))

    for i in range(num_test):

      #######################################################################

      # TODO:                                                               #

      # Compute the l2 distance between the ith test point and all training #

      # points, and store the result in dists[i, :].                        #

      #######################################################################

      

      dists[i,:] = np.linalg.norm(X[i,:] - self.X_train, axis=1)  

        

      #######################################################################

      #                         END OF YOUR CODE                            #

      #######################################################################

    return dists



  def compute_distances_no_loops(self, X):

    """

    Compute the distance between each test point in X and each training point

    in self.X_train using no explicit loops.



    Input / Output: Same as compute_distances_two_loops

    """

    num_test = X.shape[0]

    num_train = self.X_train.shape[0]

    dists = np.zeros((num_test, num_train)) 

    #########################################################################

    # TODO:                                                                 #

    # Compute the l2 distance between all test points and all training      #

    # points without using any explicit loops, and store the result in      #

    # dists.                                                                #

    #                                                                       #

    # You should implement this function using only basic array operations; #

    # in particular you should not use functions from scipy.                #

    #                                                                       #

    # HINT: Try to formulate the l2 distance using matrix multiplication    #

    #       and two broadcast sums.                                         #

    #########################################################################

    

    X_norms = np.sum( X**2, axis=1).reshape(-1,1)

    

    X_train_norms = np.sum(self.X_train ** 2,axis=1)

    

    dists = (X_norms + X_train_norms - 2 * X.dot(self.X_train.T)) ** .5

    #########################################################################

    #                         END OF YOUR CODE                              #

    #########################################################################

    return dists



  def predict_labels(self, dists, k=1):

    """

    Given a matrix of distances between test points and training points,

    predict a label for each test point.



    Inputs:

    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]

      gives the distance betwen the ith test point and the jth training point.



    Returns:

    - y: A numpy array of shape (num_test,) containing predicted labels for the

      test data, where y[i] is the predicted label for the test point X[i].  

    """

    num_test = dists.shape[0]

    y_pred = np.zeros(num_test)

    for i in range(num_test):

      # A list of length k storing the labels of the k nearest neighbors to

      # the ith test point.

      

      #########################################################################

      # TODO:                                                                 #

      # Use the distance matrix to find the k nearest neighbors of the ith    #

      # testing point, and use self.y_train to find the labels of these       #

      # neighbors. Store these labels in closest_y.                           #

      # Hint: Look up the function numpy.argsort.                             #

      #########################################################################

      closest_y = np.argsort(dists[i,:])[:k]

      #########################################################################

      # TODO:                                                                 #

      # Now that you have found the labels of the k nearest neighbors, you    #

      # need to find the most common label in the list closest_y of labels.   #

      # Store this label in y_pred[i]. Break ties by choosing the smaller     #

      # label.                                                                #

      #########################################################################

      

      neighbor_labels = self.y_train[closest_y]

      label_counts = np.bincount(neighbor_labels)

      y_pred[i] = np.argmax(label_counts)

    

      #########################################################################

      #                           END OF YOUR CODE                            # 

      #########################################################################



    return y_pred
#from cs231n.classifiers import KNearestNeighbor



# Create a kNN classifier instance. 

# Remember that training a kNN classifier is a noop: 

# the Classifier simply remembers the data and does no further processing 

classifier = KNearestNeighbor()

classifier.train(X_train, y_train)
# Open cs231n/classifiers/k_nearest_neighbor.py and implement

# compute_distances_two_loops.



# Test your implementation:

dists = classifier.compute_distances_two_loops(X_test)

print(dists.shape)
# We can visualize the distance matrix: each row is a single test example and

# its distances to training examples

plt.imshow(dists, interpolation='none')

plt.show()
# Now implement the function predict_labels and run the code below:

# We use k = 1 (which is Nearest Neighbor).

y_test_pred = classifier.predict_labels(dists, k=1)



# Compute and print the fraction of correctly predicted examples

num_correct = np.sum(y_test_pred == y_test)

accuracy = float(num_correct) / num_test

print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
y_test_pred = classifier.predict_labels(dists, k=5)

num_correct = np.sum(y_test_pred == y_test)

accuracy = float(num_correct) / num_test

print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
# Now lets speed up distance matrix computation by using partial vectorization

# with one loop. Implement the function compute_distances_one_loop and run the

# code below:

dists_one = classifier.compute_distances_one_loop(X_test)



# To ensure that our vectorized implementation is correct, we make sure that it

# agrees with the naive implementation. There are many ways to decide whether

# two matrices are similar; one of the simplest is the Frobenius norm. In case

# you haven't seen it before, the Frobenius norm of two matrices is the square

# root of the squared sum of differences of all elements; in other words, reshape

# the matrices into vectors and compute the Euclidean distance between them.

difference = np.linalg.norm(dists - dists_one, ord='fro')

print('One loop difference was: %f' % (difference, ))

if difference < 0.001:

    print('Good! The distance matrices are the same')

else:

    print('Uh-oh! The distance matrices are different')
# Now implement the fully vectorized version inside compute_distances_no_loops

# and run the code

dists_two = classifier.compute_distances_no_loops(X_test)



# check that the distance matrix agrees with the one we computed before:

difference = np.linalg.norm(dists - dists_two, ord='fro')

print('No loop difference was: %f' % (difference, ))

if difference < 0.001:

    print('Good! The distance matrices are the same')

else:

    print('Uh-oh! The distance matrices are different')
# Let's compare how fast the implementations are

def time_function(f, *args):

    """

    Call a function f with args and return the time (in seconds) that it took to execute.

    """

    import time

    tic = time.time()

    f(*args)

    toc = time.time()

    return toc - tic



two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)

print('Two loop version took %f seconds' % two_loop_time)



one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)

print('One loop version took %f seconds' % one_loop_time)



no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)

print('No loop version took %f seconds' % no_loop_time)



# You should see significantly faster performance with the fully vectorized implementation!



# NOTE: depending on what machine you're using, 

# you might not see a speedup when you go from two loops to one loop, 

# and might even see a slow-down.
num_folds = 5

for i in range(num_folds):

      print(list(set(range(num_folds+1)).difference([i,i+1])))
num_folds = 5

k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]



X_train_folds = []

y_train_folds = []

################################################################################

# TODO:                                                                        #

# Split up the training data into folds. After splitting, X_train_folds and    #

# y_train_folds should each be lists of length num_folds, where                #

# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #

# Hint: Look up the numpy array_split function.                                #

################################################################################

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



split_idx = np.array_split(range(X_train.shape[0]), num_folds)

X_train_folds = np.array([X_train[train_idx] for train_idx in split_idx])

y_train_folds = np.array([y_train[train_idx].reshape(-1,1) for train_idx in split_idx])



# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



# A dictionary holding the accuracies for different values of k that we find

# when running cross-validation. After running cross-validation,

# k_to_accuracies[k] should be a list of length num_folds giving the different

# accuracy values that we found when using that value of k.

k_to_accuracies = {}





################################################################################

# TODO:                                                                        #

# Perform k-fold cross validation to find the best value of k. For each        #

# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #

# where in each case you use all but one of the folds as training data and the #

# last fold as a validation set. Store the accuracies for all fold and all     #

# values of k in the k_to_accuracies dictionary.                               #

################################################################################

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



knn = KNearestNeighbor()



for k in tqdm_notebook(k_choices):

    accuracies = []

    

    for i in range(num_folds):

        all_other_fold_idx = list(set(range(num_folds)).difference([i]))

        X_train_cv = np.vstack(X_train_folds[all_other_fold_idx])

        y_train_cv = np.vstack(y_train_folds[all_other_fold_idx]).flatten()

        X_valid_cv = X_train_folds[i]

        y_valid_cv = y_train_folds[i].flatten()

        

        knn.train(X_train_cv,y_train_cv)

        

        knn_valid_pred = knn.predict(X_valid_cv, k=k, num_loops=0)

        

        num_correct = np.sum(knn_valid_pred == y_valid_cv)

        accuracy    = float(num_correct) / y_valid_cv.shape[0]

        

        accuracies.append(accuracy)

    

    k_to_accuracies[k] = accuracies



# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



# Print out the computed accuracies

for k in sorted(k_to_accuracies):

    for accuracy in k_to_accuracies[k]:

        print('k = %d, accuracy = %f' % (k, accuracy))
# plot the raw observations

for k in k_choices:

    accuracies = k_to_accuracies[k]

    plt.scatter([k] * len(accuracies), accuracies)



# plot the trend line with error bars that correspond to standard deviation

accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])

accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])

plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)

plt.title('Cross-validation on k')

plt.xlabel('k')

plt.ylabel('Cross-validation accuracy')

plt.show()
# Based on the cross-validation results above, choose the best value for k,   

# retrain the classifier using all the training data, and test it on the test

# data. You should be able to get above 28% accuracy on the test data.

best_k = 1



classifier = KNearestNeighbor()

classifier.train(X_train, y_train)

y_test_pred = classifier.predict(X_test, k=best_k)



# Compute and display the accuracy

num_correct = np.sum(y_test_pred == y_test)

accuracy = float(num_correct) / num_test

print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))