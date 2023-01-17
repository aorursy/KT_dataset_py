import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.python.framework import ops

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

from itertools import cycle



plt.rcParams['figure.figsize'] = (7, 4)

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'

%matplotlib inline
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

    l2_regularization_cost = cost * l2

    

    return l2_regularization_cost
def compute_cross_entropy_cost(ZL, Y):

    cross_entropy_cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = ZL, \

                                                                               labels = Y))

    

    return cross_entropy_cost                                   
def random_mini_batches(X, Y, minibatch_size = 64):

    m = X.shape[1]

    minibatches = []

    

    np.random.seed(next(seeds))

    permutation = list(np.random.permutation(m))

    shuffled_X = X[:, permutation]

    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    

    num_complete_minibatches = m // minibatch_size

    for k in range(0, num_complete_minibatches):

        minibatch_X = shuffled_X[:, k * minibatch_size : (k + 1) * minibatch_size]

        minibatch_Y = shuffled_Y[:, k * minibatch_size : (k + 1) * minibatch_size]

        minibatch = (minibatch_X, minibatch_Y)

        minibatches.append(minibatch)

    

    if m % minibatch_size != 0:

        minibatch_X = shuffled_X[:, num_complete_minibatches * minibatch_size :]

        minibatch_Y = shuffled_Y[:, num_complete_minibatches * minibatch_size :]

        minibatch = (minibatch_X, minibatch_Y)

        minibatches.append(minibatch)

        

    return minibatches
def model(X_train, Y_train, layers_dims, l2 = 1e-6, learning_rate = 0.0001, 

          num_epochs = 1500, minibatch_size = 64, print_cost = True):

    ops.reset_default_graph()

    #tf.set_random_seed(seed)

    (n_x, m) = X_train.shape

    n_y = Y_train.shape[0]

    costs = []

    

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

                _, minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, 

                                                                             Y: minibatch_Y})

                epoch_cost += minibatch_cost

                

            epoch_cost = epoch_cost / m    

            costs.append(epoch_cost)    

            

            if print_cost and epoch % 100 == 0:

                print('Cost after epoch {}: {}'.format(epoch, np.float(epoch_cost)))

        else:

            if print_cost:

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

    a = tf.sigmoid(z)

    

    with tf.Session() as sess:

        proba = sess.run(a, feed_dict = {x: X})

        

    return proba
def model_evaluation(parameters, feature_matrix, target):

    probs = predict(parameters, feature_matrix)

    (fpr, tpr, thresholds) = roc_curve(y_true = target.ravel(), y_score = probs.ravel())

    auc_score = auc(x = fpr, y = tpr)

    fig, ax = plt.subplots()

    ax.plot(fpr, tpr, 'r-', linewidth = 2)

    ax.plot([0, 1], [0, 1], 'k--', linewidth = 1)

    plt.title('ROC curve with AUC = {0:.3f}'.format(auc_score))

    plt.xlabel('fpr')

    plt.ylabel('tpr')

    plt.axis([-0.01, 1.01, -0.01, 1.01])

    plt.tight_layout()

    

    return {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': auc_score}
def k_fold_cross_validation(train_data, k, n_h, l2):

    layers_dims = [29, n_h, 1]

    fold_size = train_data.shape[0] // k

    np.random.seed(next(seeds))

    permutation = list(np.random.permutation(train_data.shape[0]))

    shuffled_data = train_data.values[permutation, :]

    shuffled_data = shuffled_data.T

 

    error = 0.0

    for i in range(k):

        val_X = shuffled_data[:-1, i * fold_size : (i + 1) * fold_size]

        val_y = shuffled_data[-1, i * fold_size : (i + 1) * fold_size].reshape(1, -1)

        

        train_X = np.concatenate([shuffled_data[:-1, 0 : i * fold_size],  \

                                  shuffled_data[:-1, (i + 1) * fold_size :]], axis = 1)

        train_y = np.concatenate([shuffled_data[-1, 0 : i * fold_size], \

                                 shuffled_data[-1, (i + 1) * fold_size :]])

        train_y = train_y.reshape(1, -1)

        

        parameters, _ = model(train_X, train_y, layers_dims, l2, learning_rate = 1e-4, \

                                   num_epochs = 1500, minibatch_size = 16, print_cost = False)

        

        probs = predict(parameters, val_X)

        preds = np.where(probs > 0.5, 1, 0)

        error += np.sum(preds != val_y)

        

    accuracy = 1.0 - error / train_data.shape[0]

    return accuracy 
def tune_hparams(train_data, hparams):

    n_h = hparams['n_h'] 

    l2 = hparams['l2'] 

    accuracies = []

    for n in range(len(n_h)):

        accuracy = k_fold_cross_validation(train_data, 5, n_h[n], l2[n])

        accuracies.append(accuracy)

        print('Trial = {}, n_h = {}, l2 = {}, accuracy = {}'.format(n, n_h[n], l2[n], accuracy))

    

    return accuracies 
data = pd.read_csv('../input/creditcard.csv')
features = data.columns

features = [str(s) for s in features]

label = features[-1]

features = features[1 : -1]

data = data[features + [label]]
scaler = StandardScaler().fit(data[features])

scaler_mean = scaler.mean_

scaler_scale = scaler.scale_

data[features] = scaler.transform(data[features])
train_data, test_data = train_test_split(data, test_size = 0.2, random_state = 1)
np.random.seed(1)

train_positive = train_data[train_data[label] == 1]

train_negative = train_data[train_data[label] == 0]

indices = np.random.choice(a = train_negative.index, size = train_positive.shape[0], replace = False)

sample_negative = train_negative.loc[indices, :]

sample = pd.concat([train_positive, train_negative.loc[indices, :]], axis = 0)
np.random.seed(1)

seeds = np.random.randint(0, 10000, 10000)

seeds = cycle(seeds)
parameters, costs = model(sample[features].values.T, sample[label].values.reshape(1, -1), [29, 10, 1], 

                          l2 = 0.001, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 16, 

                          print_cost = True)
plt.plot(costs)

plt.xlabel('epoch')

plt.ylabel('cost')

plt.title('Cost function minimizes during a few epochs.')

plt.tight_layout()
#n_h = np.arange(4, 22, 2)

#l2 = 10 ** np.linspace(-3, 3, 7)

#n_h, l2 = np.meshgrid(n_h, l2)

#hparams = {'n_h': n_h.ravel(), 'l2': l2.ravel()}

#accuracies = tune_hparams(sample, hparams)                
#n_h = np.ones(21) * 16

#l2 = 10 ** np.linspace(-2, 2, 21)

#hparams = {'n_h': n_h, 'l2': l2}

#accuracies = tune_hparams(sample, hparams)     
parameters, costs = model(sample[features].values.T, sample[label].values.reshape(1, -1), [29, 16, 1], 

                          l2 = 0.25, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 16, 

                          print_cost = True)
metrics = model_evaluation(parameters, test_data[features].T, test_data[label])
plt.plot(metrics['thresholds'], metrics['tpr'], 'r-', linewidth = 2, label = 'tpr')

plt.plot(metrics['thresholds'], metrics['fpr'], 'b-', linewidth = 2, label = 'fpr')

plt.legend(loc = 'best')

plt.axis([0, 1, 0, 1])

plt.xlabel('threshold')

plt.tight_layout()
probs = predict(parameters, test_data[features].T)

preds = np.where(probs > 0.9, 1, 0)

tn, fp, fn, tp = confusion_matrix(y_true = test_data[label].values.ravel(), \

                                  y_pred = preds.ravel()).ravel()

print ('(tn, fp, fn, tp) = ({}, {}, {}, {})'.format(tn, fp, fn, tp))

print ('precision = {}'.format(tp / (tp + fp)))

print ('recall = {}'.format(tp / (tp + fn)))

print ('accuracy = {}'.format((tp + tn) / float(len(test_data))))