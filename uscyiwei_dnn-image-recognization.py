import os

import time

import sys

import numpy as np

import h5py

import matplotlib.pyplot as plt

import scipy

import random

import pandas as pd

import warnings



from PIL import Image

from scipy import ndimage

from itertools import chain

from skimage.io import imread, imshow, imread_collection, concatenate_images

from skimage.transform import resize

from skimage.morphology import label

from keras.models import Model, load_model

from keras.layers import Input

from keras.layers.core import Lambda

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K

from tqdm import tqdm



%matplotlib inline

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'



%load_ext autoreload

%autoreload 2



warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
# Set some parameters

IMG_WIDTH = 128

IMG_HEIGHT = 128

IMG_CHANNELS = 3

TRAIN_PATH = '../input/stage1_train/'

TEST_PATH = '../input/stage1_test/'
# Get train and test IDs

train_ids = next(os.walk(TRAIN_PATH))[1]

test_ids = next(os.walk(TEST_PATH))[1]
# Get and resize train images and masks

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

print('Getting and resizing train images and masks ... ')

sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):

    path = TRAIN_PATH + id_

    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_train[n] = img

    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    for mask_file in next(os.walk(path + '/masks/'))[2]:

        mask_ = imread(path + '/masks/' + mask_file)

        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 

                                      preserve_range=True), axis=-1)

        mask = np.maximum(mask, mask_)

    Y_train[n] = mask



# Get and resize test images

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

sizes_test = []

print('Getting and resizing test images ... ')

sys.stdout.flush()

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):

    path = TEST_PATH + id_

    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]

    sizes_test.append([img.shape[0], img.shape[1]])

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_test[n] = img



print('Done!')
# Check if training data looks all right

ix = random.randint(0, len(train_ids))

imshow(X_train[ix])

plt.show()

imshow(np.squeeze(Y_train[ix]))

plt.show()
# check the size of dataset 

m_train = X_train.shape[0]

num_px = X_train.shape[1]

m_test = X_test.shape[0]



print ("Number of training examples: " + str(m_train))

print ("Number of testing examples: " + str(m_test))

print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")

print ("train_x_orig shape: " + str(X_train.shape))

print ("train_y shape: " + str(Y_train.shape))

print ("test_x_orig shape: " + str(X_test.shape))
# Reshape the training and test examples 

train_x_flatten = X_train.reshape(X_train.shape[0], -1).T # The "-1" makes reshape flatten the remaining dimensions 

test_x_flatten = X_test.reshape(X_test.shape[0], -1).T

    

# Standardize data to have feature values between 0 and 1.

train_x = train_x_flatten/255.

test_x = test_x_flatten/255.



print ("train_x's shape: " + str(train_x.shape))

print ("test_x's shape: " + str(test_x.shape))
# FUNCTION: sigmoid

def sigmoid(x):

    

    s = 1/(1+np.exp(-x))

    

    return s



# FUNCTION: relu

def relu(x):

    s = max(0,x)

    return s



# FUNCTION: initialize_parameters

def initialize_parameters(n_x, n_h, n_y):

    W1 = np.random.randn(n_h,n_x)*0.01

    b1 = np.zeros((n_h, 1))

    W2 = np.random.randn(n_y,n_h)*0.01

    b2 = np.zeros((n_y,1))

    assert(W1.shape == (n_h, n_x))

    assert(b1.shape == (n_h, 1))

    assert(W2.shape == (n_y, n_h))

    assert(b2.shape == (n_y, 1))

    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2}

    return parameters    



# FUNCTION: linear_forward

def linear_forward(A, W, b):

    Z = np.dot(W,A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))

    cache = (A, W, b)

    return Z, cache



# FUNCTION: compute_cost

def compute_cost(AL, Y):

    m = Y.shape[1]

    cost = -1/m* np.sum(np.multiply(np.log(AL),Y) + np.multiply(np.log(1 - AL),1 - Y))

    cost = np.squeeze(cost)    

    assert(cost.shape == ())

    return cost



# FUNCTION: linear_backward



def linear_backward(dZ, cache):

    A_prev, W, b = cache

    m = A_prev.shape[1]

    dW = 1/m * np.dot(dZ,A_prev.T)

    db = np.matrix(1/m * np.sum(dZ))

    dA_prev = np.dot(W.T,dZ)

    assert (dA_prev.shape == A_prev.shape)

    assert (dW.shape == W.shape)

    assert (db.shape == b.shape)

    return dA_prev, dW, db



def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2 

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    dW1 = grads['dW1']

    db1 = grads['db1']

    dW2 = grads['dW2']

    db2 = grads['db2']

    W1 = W1 - learning_rate * dW1

    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2

    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2}

    return parameters
### CONSTANTS DEFINING THE MODEL ####

n_x = 49152     # num_px * num_px * 3

n_h = 7

n_y = 1

layers_dims = (n_x, n_h, n_y)
# FUNCTION: two_layer_model



def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):

    

    grads = {}

    costs = []                              

    m = X.shape[1]                           

    (n_x, n_h, n_y) = layers_dims

    

    parameters = initialize_parameters(n_x, n_h, n_y)

    

    W1 = parameters["W1"]

    b1 = parameters["b1"]

    W2 = parameters["W2"]

    b2 = parameters["b2"]

    

    for i in range(0, num_iterations):



        Z1, linear_cache1 = linear_forward(X, W1, b1)

        A1,activation_cache1 = relu(Z1)

        cache1 = (linear_cache1, activation_cache1)

        Z2, linear_cache2 =  linear_forward(A1, W2, b2)

        A2, activation_cache2 = sigmoid(Z2)

        cache2 = (linear_cache2, activation_cache2)

        

        cost = compute_cost(A2, Y)

    

        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        

        #......

        linear_cache2, activation_cache2 = cache2

        dZ1 = sigmoid_backward(dA2, activation_cache2)

        dA1, dW2, db2 = linear_backward(dZ1, linear_cache2)

        

        linear_cache1, activation_cache1 = cache1

        dZ2 = relu_backward(dA1, activation_cache1)

        dA0, dW1, db1 = linear_backward(dZ2, linear_cache1)

        

        

        grads['dW1'] = dW1

        grads['db1'] = db1

        grads['dW2'] = dW2

        grads['db2'] = db2

        



        parameters = update_parameters(parameters, grads, learning_rate)



        W1 = parameters["W1"]

        b1 = parameters["b1"]

        W2 = parameters["W2"]

        b2 = parameters["b2"]

        

        if print_cost and i % 100 == 0:

            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))

        if print_cost and i % 100 == 0:

            costs.append(cost)

       



    plt.plot(np.squeeze(costs))

    plt.ylabel('cost')

    plt.xlabel('iterations (per tens)')

    plt.title("Learning rate =" + str(learning_rate))

    plt.show()

    

    return parameters
# Load dataset and start training



train_subset = 10000  

  

graph = tf.Graph()  

with graph.as_default():  

    # Input data.                    

    # Load the training, validation and test data into constants that are  

    # attached to the graph.  

    tf_train_dataset = tf.constant(train_x[:train_subset, :])  

    tf_train_labels = tf.constant(train_labels[:train_subset])  

      

    tf_valid_dataset = tf.constant(valid_dataset)  

    tf_test_dataset = tf.constant(test_dataset)  

    

    # Variables.定义变量 要训练得到的参数weight, bias  ----------------------------------------2  

    # These are the parameters that we are going to be training. The weight  

    # matrix will be initialized using random values following a (truncated)  

    # normal distribution. The biases get initialized to zero.  

    weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels])) # changing when training   

    biases = tf.Variable(tf.zeros([num_labels])) # changing when training   

      

    #   tf.truncated_normal  

    #   tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)  

    #   Outputs random values from a truncated normal distribution.  

    #  The generated values follow a normal distribution with specified mean and  

    #  standard deviation, except that values whose magnitude is more than 2 standard  

    #  deviations from the mean are dropped and re-picked.  

      

    # tf.zeros  

    #  tf.zeros([10])      <tf.Tensor 'zeros:0' shape=(10,) dtype=float32>  

  

  

    

    # Training computation. 训练数据                                ----------------------------------------3  

    # We multiply the inputs with the weight matrix, and add biases. We compute  

    # the softmax and cross-entropy (it's one operation in TensorFlow, because  

    # it's very common, and it can be optimized). We take the average of this  

    # cross-entropy across all training examples: that's our loss.  

    logits = tf.matmul(tf_train_dataset, weights) + biases             # tf.matmul          matrix multiply       

      

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))  # compute average cross entropy loss  

    #  softmax_cross_entropy_with_logits  

      

    # The activation ops provide different types of nonlinearities for use in neural  

    # networks.  These include smooth nonlinearities (`sigmoid`, `tanh`, `elu`,  

    #   `softplus`, and `softsign`), continuous but not everywhere differentiable  

    # functions (`relu`, `relu6`, and `relu_x`), and random regularization (`dropout`).  

      

      

    #  tf.reduce_mean  

    #    tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)  

    #   Computes the mean of elements across dimensions of a tensor.  

    

    # Optimizer.                                                                    -----------------------------------------4  

    # We are going to find the minimum of this loss using gradient descent.  

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)     # 0.5 means learning rate  

    #  tf.train.GradientDescentOptimizer(  

    #  tf.train.GradientDescentOptimizer(self, learning_rate, use_locking=False, name='GradientDescent')  

      

      

      

    

    # Predictions for the training, validation, and test data.---------------------------------------5  

    # These are not part of training, but merely here so that we can report  

    # accuracy figures as we train.  

      

    train_prediction = tf.nn.softmax(logits) # weights  and bias have been changed  

    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)  

    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)  

      

    # tf.nn.softmax  

    #  Returns: A `Tensor`. Has the same type as `logits`. Same shape as `logits`.(num, 784) *(784,10)  + = (num, 10)  



    