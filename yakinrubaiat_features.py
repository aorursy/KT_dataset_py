path = '../input/cs231n/'
import os

os.chdir(path)
import random

import numpy as np

from cs231n.data_utils import load_CIFAR10

import matplotlib.pyplot as plt





%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'



# for auto-reloading extenrnal modules

# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

%load_ext autoreload

%autoreload 2
from cs231n.features import color_histogram_hsv, hog_feature



def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):

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

    

    # Subsample the data

    mask = list(range(num_training, num_training + num_validation))

    X_val = X_train[mask]

    y_val = y_train[mask]

    mask = list(range(num_training))

    X_train = X_train[mask]

    y_train = y_train[mask]

    mask = list(range(num_test))

    X_test = X_test[mask]

    y_test = y_test[mask]

    

    return X_train, y_train, X_val, y_val, X_test, y_test



X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
from cs231n.features import *



num_color_bins = 10 # Number of bins in the color histogram

feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]

X_train_feats = extract_features(X_train, feature_fns, verbose=True)

X_val_feats = extract_features(X_val, feature_fns)

X_test_feats = extract_features(X_test, feature_fns)



# Preprocessing: Subtract the mean feature

mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)

X_train_feats -= mean_feat

X_val_feats -= mean_feat

X_test_feats -= mean_feat



# Preprocessing: Divide by standard deviation. This ensures that each feature

# has roughly the same scale.

std_feat = np.std(X_train_feats, axis=0, keepdims=True)

X_train_feats /= std_feat

X_val_feats /= std_feat

X_test_feats /= std_feat



# Preprocessing: Add a bias dimension

X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])

X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])

X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])
def svm_loss_vectorized(W, X, y, reg):

    """

    Structured SVM loss function, vectorized implementation.



    Inputs and outputs are the same as svm_loss_naive.

    """

    loss = 0.0

    dW = np.zeros(W.shape) # initialize the gradient as zero



    #############################################################################

    # TODO:                                                                     #

    # Implement a vectorized version of the structured SVM loss, storing the    #

    # result in loss.                                                           #

    #############################################################################

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    

    num_train = X.shape[0]

    

    scores = X.dot(W)

    

    correct_class_scores = scores[range(num_train),y].reshape(-1,1)

    

    #print(correct_class_scores.shape)

    

    margins = scores - correct_class_scores + 1

    

    #print(margins.clip(min=0).sum().shape)

    

    loss  = margins.clip(min=0).sum() - num_train  # We add 1 with the correct class also

    

    loss /= num_train

    loss += reg* np.sum(W*W)

    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



    #############################################################################

    # TODO:                                                                     #

    # Implement a vectorized version of the gradient for the structured SVM     #

    # loss, storing the result in dW.                                           #

    #                                                                           #

    # Hint: Instead of computing the gradient from scratch, it may be easier    #

    # to reuse some of the intermediate values that you used to compute the     #

    # loss.                                                                     #

    #############################################################################

    

    idx_positive_margins = np.greater(margins,0).astype('int')

    idx_positive_margins[range(num_train),y] -= idx_positive_margins.sum(axis=1)

    

    dW = X.T.dot(idx_positive_margins)

    dW /= num_train

    dW += 2 * reg * W



    return loss, dW
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

            idx = np.random.choice(range(num_train),size=batch_size)

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

        #y_pred = np.zeros(X.shape[0])

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
class LinearSVM(LinearClassifier):

    """ A subclass that uses the Multiclass SVM loss function """



    def loss(self, X_batch, y_batch, reg):

        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)
from tqdm import tqdm_notebook
# Use the validation set to tune the learning rate and regularization strength



#from cs231n.classifiers.linear_classifier import LinearSVM



learning_rates = [1e-9, 1e-8, 1e-7]

regularization_strengths = [5e4, 5e5, 5e6]



results = {}

best_val = -1

best_svm = None



################################################################################

# TODO:                                                                        #

# Use the validation set to set the learning rate and regularization strength. #

# This should be identical to the validation that you did for the SVM; save    #

# the best trained classifer in best_svm. You might also want to play          #

# with different numbers of bins in the color histogram. If you are careful    #

# you should be able to get accuracy of near 0.44 on the validation set.       #

################################################################################

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



for lr in tqdm_notebook(learning_rates):

    for reg in tqdm_notebook(regularization_strengths):

        svm = LinearSVM()

        _=svm.train(X_train_feats, y_train,learning_rate=lr,

                   reg=reg,num_iters=1500,verbose=False)

        y_train_pred = svm.predict(X_train_feats)

        train_acc = np.mean(y_train==y_train_pred)

        y_val_pred = svm.predict(X_val_feats)

        val_acc = np.mean(y_val == y_val_pred)

        results[(lr,reg)] = (train_acc,val_acc)

        if val_acc > best_val:

            best_val = val_acc

            best_svm = svm



# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



# Print out results.

for lr, reg in sorted(results):

    train_accuracy, val_accuracy = results[(lr, reg)]

    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (

                lr, reg, train_accuracy, val_accuracy))

    

print('best validation accuracy achieved during cross-validation: %f' % best_val)
# Evaluate your trained SVM on the test set

y_test_pred = best_svm.predict(X_test_feats)

test_accuracy = np.mean(y_test == y_test_pred)

print(test_accuracy)
# An important way to gain intuition about how an algorithm works is to

# visualize the mistakes that it makes. In this visualization, we show examples

# of images that are misclassified by our current system. The first column

# shows images that our system labeled as "plane" but whose true label is

# something other than "plane".



examples_per_class = 8

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for cls, cls_name in enumerate(classes):

    idxs = np.where((y_test != cls) & (y_test_pred == cls))[0]

    idxs = np.random.choice(idxs, examples_per_class, replace=False)

    for i, idx in enumerate(idxs):

        plt.subplot(examples_per_class, len(classes), i * len(classes) + cls + 1)

        plt.imshow(X_test[idx].astype('uint8'))

        plt.axis('off')

        if i == 0:

            plt.title(cls_name)

plt.show()
# Preprocessing: Remove the bias dimension

# Make sure to run this cell only ONCE

print(X_train_feats.shape)

X_train_feats = X_train_feats[:, :-1]

X_val_feats = X_val_feats[:, :-1]

X_test_feats = X_test_feats[:, :-1]



print(X_train_feats.shape)
class TwoLayerNet(object):

    """

    A two-layer fully-connected neural network. The net has an input dimension of

    N, a hidden layer dimension of H, and performs classification over C classes.

    We train the network with a softmax loss function and L2 regularization on the

    weight matrices. The network uses a ReLU nonlinearity after the first fully

    connected layer.



    In other words, the network has the following architecture:



    input - fully connected layer - ReLU - fully connected layer - softmax



    The outputs of the second fully-connected layer are the scores for each class.

    """



    def __init__(self, input_size, hidden_size, output_size, std=1e-4):

        """

        Initialize the model. Weights are initialized to small random values and

        biases are initialized to zero. Weights and biases are stored in the

        variable self.params, which is a dictionary with the following keys:



        W1: First layer weights; has shape (D, H)

        b1: First layer biases; has shape (H,)

        W2: Second layer weights; has shape (H, C)

        b2: Second layer biases; has shape (C,)



        Inputs:

        - input_size: The dimension D of the input data.

        - hidden_size: The number of neurons H in the hidden layer.

        - output_size: The number of classes C.

        """

        self.params = {}

        self.params['W1'] = std * np.random.randn(input_size, hidden_size)

        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = std * np.random.randn(hidden_size, output_size)

        self.params['b2'] = np.zeros(output_size)



    def loss(self, X, y=None, reg=0.0):

        """

        Compute the loss and gradients for a two layer fully connected neural

        network.



        Inputs:

        - X: Input data of shape (N, D). Each X[i] is a training sample.

        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is

          an integer in the range 0 <= y[i] < C. This parameter is optional; if it

          is not passed then we only return scores, and if it is passed then we

          instead return the loss and gradients.

        - reg: Regularization strength.



        Returns:

        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is

        the score for class c on input X[i].



        If y is not None, instead return a tuple of:

        - loss: Loss (data loss and regularization loss) for this batch of training

          samples.

        - grads: Dictionary mapping parameter names to gradients of those parameters

          with respect to the loss function; has the same keys as self.params.

        """

        # Unpack variables from the params dictionary

        W1, b1 = self.params['W1'], self.params['b1']

        W2, b2 = self.params['W2'], self.params['b2']

        N, D = X.shape



        # Compute the forward pass

        a1 = X.dot(W1) + b1

        a1_relu = np.maximum(a1,np.zeros_like(a1))

        scores = a1_relu.dot(W2) + b2

        #############################################################################

        # TODO: Perform the forward pass, computing the class scores for the input. #

        # Store the result in the scores variable, which should be an array of      #

        # shape (N, C).                                                             #

        #############################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



        #pass



        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



        # If the targets are not given then jump out, we're done

        if y is None:

            return scores



        # Compute the loss

        #loss = None

        #############################################################################

        # TODO: Finish the forward pass, and compute the loss. This should include  #

        # both the data loss and L2 regularization for W1 and W2. Store the result  #

        # in the variable loss, which should be a scalar. Use the Softmax           #

        # classifier loss.                                                          #

        #############################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



        correct_class_scores = scores[range(X.shape[0]),y].reshape(-1,1)

        max_scores =scores.max(axis=1,keepdims=True)

        scores -= max_scores

        

        loss = -correct_class_scores.sum() + max_scores.sum() + np.log(np.exp(scores).sum(axis=1)).sum()

        loss /= N

        loss += reg * (np.sum(W1*W1) + np.sum(W2*W2))



        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



        # Backward pass: compute gradients

        grads = {}

        #############################################################################

        # TODO: Compute the backward pass, computing the derivatives of the weights #

        # and biases. Store the results in the grads dictionary. For example,       #

        # grads['W1'] should store the gradient on W1, and be a matrix of same size #

        #############################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



        softmax_deriv = (np.exp(scores)/np.exp(scores).sum(axis=1).reshape(-1,1))

        softmax_deriv[range(N),y] -= 1

        

        dW2 = a1_relu.T.dot(softmax_deriv)

        dW2 /= N

        dW2 += 2 * reg * W2

        grads['W2'] = dW2

        

        db2 = np.sum(softmax_deriv,axis=0)

        db2 /= N

        grads['b2'] = db2

        

        da1_relu = softmax_deriv.dot(W2.T)

        

        da1 = da1_relu * (a1_relu>0)

        

        dW1 = X.T.dot(da1)

        dW1 /= N

        dW1 += 2 * reg * W1

        grads['W1'] = dW1

        

        db1 = np.sum(da1,axis=0)

        db1 /= N

        grads["b1"] = db1



        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



        return loss, grads



    def train(self, X, y, X_val, y_val,

              learning_rate=1e-3, learning_rate_decay=0.95,

              reg=5e-6, num_iters=100,

              batch_size=200, verbose=False):

        """

        Train this neural network using stochastic gradient descent.



        Inputs:

        - X: A numpy array of shape (N, D) giving training data.

        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that

          X[i] has label c, where 0 <= c < C.

        - X_val: A numpy array of shape (N_val, D) giving validation data.

        - y_val: A numpy array of shape (N_val,) giving validation labels.

        - learning_rate: Scalar giving learning rate for optimization.

        - learning_rate_decay: Scalar giving factor used to decay the learning rate

          after each epoch.

        - reg: Scalar giving regularization strength.

        - num_iters: Number of steps to take when optimizing.

        - batch_size: Number of training examples to use per step.

        - verbose: boolean; if true print progress during optimization.

        """

        num_train = X.shape[0]

        iterations_per_epoch = max(num_train / batch_size, 1)



        # Use SGD to optimize the parameters in self.model

        loss_history = []

        train_acc_history = []

        val_acc_history = []



        for it in range(num_iters):

            

            indexes  = np.random.choice(X.shape[0],batch_size,replace=False)

            

            X_batch = X[indexes]

            y_batch = y[indexes]



            #########################################################################

            # TODO: Create a random minibatch of training data and labels, storing  #

            # them in X_batch and y_batch respectively.                             #

            #########################################################################

            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



            # Compute loss and gradients using the current minibatch

            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)

            loss_history.append(loss)



            #########################################################################

            # TODO: Use the gradients in the grads dictionary to update the         #

            # parameters of the network (stored in the dictionary self.params)      #

            # using stochastic gradient descent. You'll need to use the gradients   #

            # stored in the grads dictionary defined above.                         #

            #########################################################################

            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



            for params_name in self.params:

                self.params[params_name] -= learning_rate * grads[params_name]



            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



            if verbose and it % 100 == 0:

                print('iteration %d / %d: loss %f' % (it, num_iters, loss))



            # Every epoch, check train and val accuracy and decay learning rate.

            if it % iterations_per_epoch == 0:

                # Check accuracy

                train_acc = (self.predict(X_batch) == y_batch).mean()

                val_acc = (self.predict(X_val) == y_val).mean()

                train_acc_history.append(train_acc)

                val_acc_history.append(val_acc)



                # Decay learning rate

                learning_rate *= learning_rate_decay



        return {

          'loss_history': loss_history,

          'train_acc_history': train_acc_history,

          'val_acc_history': val_acc_history,

        }



    def predict(self, X):

        """

        Use the trained weights of this two-layer network to predict labels for

        data points. For each data point we predict scores for each of the C

        classes, and assign each data point to the class with the highest score.



        Inputs:

        - X: A numpy array of shape (N, D) giving N D-dimensional data points to

          classify.



        Returns:

        - y_pred: A numpy array of shape (N,) giving predicted labels for each of

          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted

          to have class c, where 0 <= c < C.

        """

        #y_pred = None



        ###########################################################################

        # TODO: Implement this function; it should be VERY simple!                #

        ###########################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



        scores = self.loss(X)

        y_pred = np.argmax(scores,axis=1)



        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



        return y_pred
input_dim = X_train_feats.shape[1]

hidden_dim = 500

num_classes = 10



net = TwoLayerNet(input_dim, hidden_dim, num_classes)

best_net = None

results = {}

best_val_acc = -1

learning_rates = np.linspace(1e-1,1,5)

regularization_strengths = np.linspace(1e-4,1e-3,3)

hidden_sizes = [300,400,500]

num_iters = 1500

batch_size = 200

learning_rate_decay = 0.95

################################################################################

# TODO: Train a two-layer neural network on image features. You may want to    #

# cross-validate various parameters as in previous sections. Store your best   #

# model in the best_net variable.                                              #

################################################################################

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



best_params = None

for lr in tqdm_notebook(learning_rates):

    for reg in tqdm_notebook(regularization_strengths):

        for hidden_size in tqdm_notebook(hidden_sizes):

            print('lr: {}, reg: {}'.format(lr,reg))

            net = TwoLayerNet(input_dim,hidden_size,num_classes)

            stats = net.train(X_train_feats,y_train,X_val_feats,y_val,

                             num_iters = num_iters,batch_size=batch_size,

                             learning_rate=lr,learning_rate_decay=learning_rate_decay,

                             reg=reg,verbose=False)

            y_train_pred = net.predict(X_train_feats)

            train_acc = np.mean(y_train == y_train_pred)



            val_acc =  (net.predict(X_val_feats) == y_val).mean()

            print('Vaildation accuracy: ',val_acc)

            results[(lr,reg)] = val_acc



            if val_acc > best_val_acc:

                best_val_acc = val_acc

                best_net = net

                best_params = (lr,reg,hidden_size)





# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

best_params
best_net = TwoLayerNet(input_dim,best_params[2],num_classes) # store the best model into this 



#################################################################################

# TODO: Tune hyperparameters using the validation set. Store your best trained  #

# model in best_net.                                                            #

#                                                                               #

# To help debug your network, it may help to use visualizations similar to the  #

# ones we used above; these visualizations will have significant qualitative    #

# differences from the ones we saw above for the poorly tuned network.          #

#                                                                               #

# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #

# write code to sweep through possible combinations of hyperparameters          #

# automatically like we did on the previous exercises.                          #

#################################################################################

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



stats = best_net.train(X_train_feats,y_train,X_val_feats,y_val,

                      num_iters=3000,batch_size=batch_size,

                      learning_rate=best_params[0],learning_rate_decay=learning_rate_decay,

                      reg=best_params[1],verbose=True)
# Plot the loss function and train / validation accuracies

plt.subplot(2, 1, 1)

plt.plot(stats['loss_history'])

plt.title('Loss history')

plt.xlabel('Iteration')

plt.ylabel('Loss')



plt.subplot(2, 1, 2)

plt.plot(stats['train_acc_history'], label='train')

plt.plot(stats['val_acc_history'], label='val')

plt.title('Classification accuracy history')

plt.xlabel('Epoch')

plt.ylabel('Classification accuracy')

plt.legend()

plt.show()
# Run your best neural net classifier on the test set. You should be able

# to get more than 55% accuracy.



test_acc = (best_net.predict(X_test_feats) == y_test).mean()

print(test_acc)