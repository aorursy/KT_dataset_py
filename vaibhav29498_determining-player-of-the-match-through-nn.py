import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

%matplotlib inline
data = pd.read_csv('../input/data.csv')
data.describe()
data.info()
data=data.dropna()
data.info()
d = data.values
X = []
Y = []
for r in d:
    Y.append(r[3])
    t = [np.concatenate((r[[0, 1, 2, 4, 5, 6, 7, 8, 9]], [0, 0, 0]))]
    for i in range(1, 23):
        t.append(r[list(range(10 + 12 * (i - 1), 10 + 12 * i))])
    X.append(t)
Y = np.asarray(Y)
X = np.asarray(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 1)
y_train = label_binarize(y_train, classes=range(22))
y_test = label_binarize(y_test, classes=range(22))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
def create_placeholders():
    X = tf.placeholder(tf.float32, shape=(None, 23, 12))
    Y = tf.placeholder(tf.float32, shape=(None, 22))
    return X,Y
def initialize_parameters(layers_dims):
    L1 = len(layers_dims)
    parameters = {}
    for l in range(1, L1):
        parameters['W' + str(l)] = tf.get_variable(shape=[layers_dims[l - 1], layers_dims[l]], initializer=tf.contrib.layers.variance_scaling_initializer(seed=l), name='W' + str(l))
        parameters['b' + str(l)] = tf.get_variable(shape=[1, layers_dims[l]], initializer=tf.zeros_initializer(), name='b' + str(l))
    return parameters
def forward_propagation(X, parameters):
    L1 = len(parameters) // 2
    values = {}
    mat = X[:, 0, 1:9]
    values['Z'] = [0 for _ in range(22)]
    for p in range(22):
        values['A' + str(p) + '0'] = tf.concat((mat, X[:, p + 1]), axis=1)
        for l in range(1, L1):
            values['Z' + str(p) + str(l)] = tf.add(tf.matmul(values['A' + str(p) + str(l - 1)], parameters['W' + str(l)]), parameters['b' + str(l)])
            values['A' + str(p) + str(l)] = tf.nn.leaky_relu(values['Z' + str(p) + str(l)])
        values['Z'][p] = tf.add(tf.matmul(values['A' + str(p) + str(L1 - 1)], parameters['W' + str(L1)]), parameters['b' + str(L1)])
    values['Z'] = tf.reshape(tf.stack(values['Z'], axis=1), [-1, 22])
    return values['Z']
def compute_cost(Y, Z):
    vars = tf.trainable_variables() 
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Z, labels=tf.argmax(Y, axis=1)))
    return cost
def model(X_train, Y_train, layers_dims, learning_rate, epochs, print_costs=False):    
    tf.reset_default_graph()
    m = X_train.shape[0]
    X, Y = create_placeholders()
    layers_dims = [20] + layers_dims + [1]
    parameters = initialize_parameters(layers_dims)
    Z = forward_propagation(X, parameters)
    cost = compute_cost(Y, Z)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    costs = []
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        num = 1
        while num <= epochs:
            _, epoch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
            if print_costs and num % 1000 == 0:
                print('Cost after epoch', num, '=', epoch_cost)
            if num % 5000 == 0:
              learning_rate -= 0.001
            if print_costs and num > 15:
                costs.append(epoch_cost)
            num += 1
        
        if print_costs:
            plt.plot(np.squeeze(costs))
            plt.xlabel('Epoch number')
            plt.ylabel('Cost')
            plt.show()
        
        parameters = sess.run(parameters)
        correct_predictions = tf.equal(tf.argmax(Y, axis=1), tf.argmax(Z, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))
        print('Training Accuracy:', accuracy.eval({X: X_train, Y: Y_train}))
        
    return parameters
parameters = model(X_train, y_train, layers_dims=[12, 6, 3], epochs=10000, learning_rate=0.001, print_costs=True)
def test(X_test, Y_test, parameters):
    X = tf.placeholder(tf.float32, shape=X_test.shape)
    Z = forward_propagation(X, parameters)
    correct_predictions = tf.equal(tf.argmax(Y_test, axis=1), tf.argmax(Z, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))
    top2 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=Z, targets=tf.argmax(Y_test, axis=1), k=2), tf.float32))
    top3 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=Z, targets=tf.argmax(Y_test, axis=1), k=3), tf.float32))
    with tf.Session() as sess:
        print('Test Accuracy:', accuracy.eval({X: X_test}))
        print('Top-2 Accuracy:', top2.eval({X: X_test}))
        print('Top-3 Accuracy:', top3.eval({X: X_test}))
        return Z.eval({X: X_test})

y_pred = test(X_test, y_test, parameters)