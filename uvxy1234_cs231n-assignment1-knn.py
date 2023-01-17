import os

#print(os.listdir("../input"))
import pickle

import numpy as np



def load_cifar10_data(filename):

    with open('../input/cifar10/'+ filename, 'rb') as file:

        batch = pickle.load(file, encoding='latin1')



    features = batch['data']

    labels = batch['labels']

    return features, labels



# Load files

batch_1, labels_1 = load_cifar10_data('data_batch_1')

batch_2, labels_2 = load_cifar10_data('data_batch_2')

batch_3, labels_3 = load_cifar10_data('data_batch_3')

batch_4, labels_4 = load_cifar10_data('data_batch_4')

batch_5, labels_5 = load_cifar10_data('data_batch_5')



X_test, y_test = load_cifar10_data('test_batch')



# Merge files

X_train = np.concatenate([batch_1,batch_2,batch_3,batch_4,batch_5], 0)

y_train = np.concatenate([labels_1,labels_2,labels_3,labels_4,labels_5], 0)

y_test = np.array(y_test)
print(X_train.shape)

print(X_test.shape)
# Run some setup code for this notebook.



import random

#from cs231n.data_utils import load_CIFAR10

import matplotlib.pyplot as plt



# This is a bit of magic to make matplotlib figures appear inline in the notebook

# rather than in a new window.

%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'



# Some more magic so that the notebook will reload external python modules;

# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

#%load_ext autoreload

#%autoreload 2
# Load the raw CIFAR-10 data.

#cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'

#X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)



#Reshape flatten raw train/test sets to 32*32*3

def data_reshape(file):

    assert file.shape[1] == 3072

    dim = np.sqrt(1024).astype(int)

    r = file[:, 0:1024].reshape(file.shape[0], dim, dim, 1)

    g = file[:, 1024:2048].reshape(file.shape[0], dim, dim, 1)

    b = file[:, 2048:3072].reshape(file.shape[0], dim, dim, 1)

    return np.concatenate([r,g,b], -1)



X_train = data_reshape(X_train)

X_test = data_reshape(X_test)



# As a sanity check, we print out the size of the training and test data.

print('Training data shape: ', X_train.shape)

print('Training labels shape: ', y_train.shape)

print('Test data shape: ', X_test.shape)

print('Test labels shape: ', y_test.shape)
# Visualize some examples from the dataset.

# We show a few examples of training images from each class.

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

num_classes = len(classes)

samples_per_class = 7

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

mask = range(num_training)

X_train = X_train[mask]

y_train = y_train[mask]



num_test = 500

mask = range(num_test)

X_test = X_test[mask]

y_test = y_test[mask]
# Reshape the image data into rows

X_train = np.reshape(X_train, (X_train.shape[0], -1))

X_test = np.reshape(X_test, (X_test.shape[0], -1))

print(X_train.shape, X_test.shape)
#import the utils

!cp -r ../input/knn-util/* ./

from k_nearest_neighbor import KNearestNeighbor

# Create a kNN classifier instance. 

# Remember that training a kNN classifier is a noop: 

# the Classifier simply remembers the data and does no further processing 

classifier = KNearestNeighbor()

classifier.train(X_train/255.0, y_train)
# Open cs231n/classifiers/k_nearest_neighbor.py and implement

# compute_distances_two_loops.



# Test your implementation:

dists = classifier.compute_distances_two_loops(X_test/255.0)

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

dists_one = classifier.compute_distances_one_loop(X_test/255.0)



# To ensure that our vectorized implementation is correct, we make sure that it

# agrees with the naive implementation. There are many ways to decide whether

# two matrices are similar; one of the simplest is the Frobenius norm. In case

# you haven't seen it before, the Frobenius norm of two matrices is the square

# root of the squared sum of differences of all elements; in other words, reshape

# the matrices into vectors and compute the Euclidean distance between them.

difference = np.linalg.norm(dists - dists_one, ord='fro')

print('Difference was: %f' % (difference, ))

if difference < 0.001:

    print('Good! The distance matrices are the same')

else:

    print('Uh-oh! The distance matrices are different')
# Now implement the fully vectorized version inside compute_distances_no_loops

# and run the code

dists_two = classifier.compute_distances_no_loops(X_test/255.0)



# check that the distance matrix agrees with the one we computed before:

difference = np.linalg.norm(dists - dists_two, ord='fro')

print('Difference was: %f' % (difference, ))

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



two_loop_time = time_function(classifier.compute_distances_two_loops, X_test/255.0)

print('Two loop version took %f seconds' % two_loop_time)



one_loop_time = time_function(classifier.compute_distances_one_loop, X_test/255.0)

print('One loop version took %f seconds' % one_loop_time)



no_loop_time = time_function(classifier.compute_distances_no_loops, X_test/255.0)

print('No loop version took %f seconds' % no_loop_time)



# you should see significantly faster performance with the fully vectorized implementation
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

X_train_folds = np.array_split(X_train/255.0, num_folds)

y_train_folds = np.array_split(y_train, num_folds)

################################################################################

#                                 END OF YOUR CODE                             #

################################################################################



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

for k in k_choices:

    k_acc = []

    for i in range(num_folds):

        X_train_fold = np.concatenate(X_train_folds[:i] + X_train_folds[i+1:])

        y_train_fold = np.concatenate(y_train_folds[:i] + y_train_folds[i+1:])

        X_val_fold = X_train_folds[i]

        y_val_fold = y_train_folds[i]

        

        classifier = KNearestNeighbor()

        classifier.train(X_train_fold/255.0, y_train_fold)

        dists_fold = classifier.compute_distances_no_loops(X_val_fold/255.0)

        

        y_val_pred = classifier.predict_labels(dists_fold, k)

        num_correct = np.sum(y_val_pred == y_val_fold)

        accuracy = float(num_correct) / y_val_fold.shape[0]

        k_acc.append(accuracy)

    k_to_accuracies[k] = k_acc

    

################################################################################

#                                 END OF YOUR CODE                             #

################################################################################



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

best_k = 10



classifier = KNearestNeighbor()

classifier.train(X_train/255.0, y_train)

y_test_pred = classifier.predict(X_test/255.0, k=best_k)



# Compute and display the accuracy

num_correct = np.sum(y_test_pred == y_test)

accuracy = float(num_correct) / num_test

print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))