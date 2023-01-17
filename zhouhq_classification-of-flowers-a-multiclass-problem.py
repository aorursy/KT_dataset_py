import numpy as np

import pandas as pd

import h5py

import matplotlib.pyplot as plt

import scipy

from scipy import ndimage

import tensorflow as tf

from tensorflow.python.framework import ops

from itertools import cycle

from sklearn.model_selection import train_test_split



%matplotlib inline

plt.rcParams['figure.figsize'] = (5, 4) # set default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'
def create_placeholders(n_x, n_y):

    X = tf.placeholder(dtype = tf.float32, shape = (n_x, None), name = 'X')

    Y = tf.placeholder(dtype = tf.float32, shape = (n_y, None), name = 'Y')

    

    return X, Y



def initialize_parameters(layers_dims):

    num_layers = len(layers_dims) - 1

    parameters = {}

    for l in range(1, num_layers + 1):

        parameters['W' + str(l)] = tf.get_variable('W' + str(l), [layers_dims[l], layers_dims[l - 1]],\

                        initializer = tf.contrib.layers.xavier_initializer(seed = next(seeds)))

        parameters['b' + str(l)] = tf.get_variable('b' + str(l), [layers_dims[l], 1], \

                                                  initializer = tf.zeros_initializer())

    return parameters   



def forward_propagation(X, parameters):

    L = len(parameters) // 2

    A = X

    for l in range(1, L):

        Z = tf.add(tf.matmul(parameters['W' + str(l)], A), parameters['b' + str(l)])

        A = tf.nn.relu(Z)

    ZL = tf.add(tf.matmul(parameters['W' + str(L)], A), parameters['b' + str(L)])

    

    return ZL



def compute_l2_regularization_cost(parameters, l2):

    L = len(parameters) // 2

    cost = 0.0

    for l in range(1, L + 1):

        cost += tf.reduce_sum(tf.nn.l2_loss(parameters['W' + str(l)]))

    l2_regularization = cost * l2

    

    return l2_regularization



def compute_cross_entropy_cost(ZL, Y):

    logits = tf.transpose(ZL)

    labels = tf.transpose(Y)

    cross_entropy_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\

                                            logits = logits, labels = labels))

    

    return cross_entropy_cost



def random_mini_batches(X, Y, minibatch_size = 64):

    m = X.shape[1]

    minibatches = []

    

    np.random.seed(next(seeds))

    permutation = list(np.random.permutation(m))

    shuffled_X = X[:, permutation]

    shuffled_Y = Y[:, permutation]

    

    num_complete_minibatches = m // minibatch_size

    for k in range(0, num_complete_minibatches):

        minibatch_X = shuffled_X[:, k * minibatch_size : (k + 1) * minibatch_size]

        minibatch_Y = shuffled_Y[:, k * minibatch_size : (k + 1) * minibatch_size]

        minibatch = (minibatch_X, minibatch_Y)

        minibatches.append(minibatch)

        

    if m % minibatch_size != 0:

        minibatch_X = shuffled_X[:, num_complete_minibatches * minibatch_size : ]

        minibatch_Y = shuffled_Y[:, num_complete_minibatches * minibatch_size : ]

        minibatch = (minibatch_X, minibatch_Y)

        minibatches.append(minibatch)

    

    return minibatches

                     

def model(X_train, Y_train, layers_dims, l2 = 1e-6, learning_rate = 0.0001, num_epochs = 100, \

         minibatch_size = 64, print_cost_interval = None):

    

    ops.reset_default_graph()

    (n_x, m) = X_train.shape

    n_y = Y_train.shape[0]

    costs =[]

    

    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters(layers_dims)

    ZL = forward_propagation(X, parameters)

    cross_entropy_cost = compute_cross_entropy_cost(ZL, Y)

    l2_regularization_cost = compute_l2_regularization_cost(parameters, l2)

    cost = cross_entropy_cost + l2_regularization_cost

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    

    with tf.Session() as sess:

        sess.run(init)

        

        for epoch in range(num_epochs):

            epoch_cost = 0.0

            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            num_minibatches = len(minibatches)

            

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch

                _, minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, \

                                                                             Y: minibatch_Y})

                epoch_cost = epoch_cost + minibatch_cost / num_minibatches 

                

            costs.append(epoch_cost)

            

            if print_cost_interval is not None and epoch % print_cost_interval == 0:

                print('Cost after epoch {}: {}'.format(epoch, np.float(epoch_cost)))

        else:

            if print_cost_interval is not None:

                print('Cost after epoch {}: {}'.format(epoch, np.float(epoch_cost)))

            

        parameters = sess.run(parameters)

    

    return parameters, costs
def predict(parameters, X):

    nx = X.shape[0]

    params = {}

    L = len(parameters) // 2

    for l in range(1, L+1):

        params['W' + str(l)] = tf.convert_to_tensor(parameters['W' + str(l)])

        params['b' + str(l)] = tf.convert_to_tensor(parameters['b' + str(l)])

    

    x = tf.placeholder(dtype = tf.float32, shape = (nx, None))

    z = forward_propagation(x, params) 

    preds = tf.argmax(z)

    

    with tf.Session() as sess:

        preds = sess.run(preds, feed_dict = {x: X})

        

    return preds
images = []

for i in range(1, 211):

    fname = '../input/flower_images/' + str(i).zfill(4) + '.png'

    image = np.array(ndimage.imread(fname, flatten = False))

    image = scipy.misc.imresize(image, size = (128, 128))

    images.append(image)

    

images = np.asarray(images)



labels = pd.read_csv('../input/flower_images/flower_labels.csv')

labels = labels['label']

labels = np.asarray(labels)
"""

with h5py.File('../input/FlowerColorImages.h5', 'w') as f:

    f.create_dataset('images', data = images)

    f.create_dataset('labels', data = labels)



with h5py.File('../input/FlowerColorImages.h5', 'r') as f:

    images = f['images'].value

    labels = f['labels'].value    

"""
images_flip = np.zeros(images.shape)

labels_flip = np.zeros(labels.shape)

for i in range(len(images)):

    images_flip[i, :, :, :] = images[i, :, np.arange(127, -1, -1), :]

    labels_flip[i] = labels[i]

images_expanded = np.concatenate([images, images_flip], axis = 0)

labels_expanded = np.concatenate([labels, labels_flip]).astype(int)
X = images_expanded[:, :, :, :3].reshape(420, -1) / 255.

Y = np.eye(10)[labels_expanded.reshape(-1)]

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.3, random_state = 1)
"""

def k_fold_cross_validation(X, Y, k, layer_dims, l2):

    fold_size = X.shape[0] // k

    np.random.seed(next(seeds))

    permutation = list(np.random.permutation(X.shape[0]))

    shuffled_X = X[permutation, :]

    shuffled_Y = Y[permutation, :]



    accuracy = 0.0

    for i in range(k):

        val_X = shuffled_X[i * fold_size : (i + 1) * fold_size, :]

        val_Y = shuffled_Y[i * fold_size : (i + 1) * fold_size, :]

        

        train_X = np.concatenate([shuffled_X[0 : i * fold_size, :],  \

                                  shuffled_X[(i + 1) * fold_size : -1, :]], axis = 0)

        train_Y = np.concatenate([shuffled_Y[0 : i * fold_size, :], \

                                 shuffled_Y[(i + 1) * fold_size : -1, :]])

        

        parameters, _ = model(train_X.T, train_Y.T, layer_dims, l2, learning_rate = 1e-4, \

                                   num_epochs = 600, minibatch_size = 32, print_cost_interval = None)

        

        preds = predict(parameters, val_X.T)

        accuracy += np.sum(preds == np.argmax(val_Y, axis = 1))

           

    accuracy = accuracy / float(k * fold_size)

    

    return accuracy 

"""
"""

np.random.seed(1)

seeds = np.random.randint(0, 10000, 10000)

seeds = cycle(seeds)

accuracies = []



for l2 in 10 ** np.linspace(-5, 1, 7):

    accuracy = k_fold_cross_validation(train_X, train_Y, 5, [49152, 64, 30, 10], l2, next(seeds))

    accuracies.append(accuracy)

    print('l2 = {0}, accuracy = {1}'.format(l2, accuracy))

    

pd.DataFrame({'l2': 10 ** np.linspace(-3, 1, 5), 'accuracy': accuracies}).to_csv( \

        'cv_cache.csv', header = True, index = False)    

"""
np.random.seed(1)

seeds = np.random.randint(0, 10000, 10000)

seeds = cycle(seeds)



parameters, costs = model(train_X.T, train_Y.T, [49152, 10], l2 = 0.001, learning_rate = 0.0001, \

                          num_epochs = 400, minibatch_size = 32, print_cost_interval = 100)
fig, ax = plt.subplots()

plt.plot(costs)

plt.xlabel('epoch')

plt.ylabel('cost')

plt.title('Cost minimization')
preds = predict(parameters, test_X.T)

accuracy = np.sum(preds == np.argmax(test_Y, axis = 1)) / float(test_Y.shape[0])

print ('Model accuracy = {}'.format(accuracy))
parameters, costs = model(train_X.T, train_Y.T, [49152, 64, 30, 10], l2 = 0.1, learning_rate = 0.0001, \

                          num_epochs = 600, minibatch_size = 32, print_cost_interval = 100)
fig, ax = plt.subplots()

plt.plot(costs)

plt.xlabel('epoch')

plt.ylabel('cost')

plt.title('Cost minimization')
preds = predict(parameters, test_X.T)

accuracy = np.sum(preds == np.argmax(test_Y, axis = 1)) / float(test_Y.shape[0])

print ('Model accuracy = {}'.format(accuracy))   