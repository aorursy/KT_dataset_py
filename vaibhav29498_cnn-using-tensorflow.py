import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
X_train = []
Y_train = []
X_test = []

with open('../input/train.csv', 'r') as train_file:
    reader = csv.reader(train_file)
    next(reader)
    for row in reader:
        Y_train.append(int(row[0]))
        X_train.append(list(map(int, row[1:])))

with open('../input/test.csv', 'r') as test_file:
    reader = csv.reader(test_file)
    next(reader)
    for row in reader:
        X_test.append(list(map(int, row)))

X_train = np.array(X_train)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = np.array(X_test)
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
Y_train = np.eye(10)[Y_train]

print('Shape of X_train:', X_train.shape)
print('Shape of Y_train:', Y_train.shape)
print('Shape of X_test:', X_test.shape)
def create_placeholders(n_h, n_w, n_c, n_y):
    X = tf.placeholder(shape=(None, n_h, n_w, n_c), dtype=tf.float32)
    Y = tf.placeholder(shape=(None, n_y), dtype=tf.int32)
    return (X, Y)
def initialize_parameters(layers):
    l = len(layers)
    parameters = {}
    for i in range(l):
        parameters['W' + str(i + 1)] = tf.get_variable(name='W' + str(i + 1), shape=layers[i], initializer=tf.contrib.layers.xavier_initializer())
    return parameters
def forward_propagation(X, parameters):
    A_prev = X
    l = len(parameters)
    for i in range(l):
        Z = tf.nn.conv2d(A_prev, parameters['W' + str(i + 1)], strides=[1, 1, 1, 1], padding='SAME')
        A = tf.nn.relu(Z)
        P = tf.nn.max_pool(A, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='SAME')
        A_prev = P
    P = tf.contrib.layers.flatten(P)
    Zl = tf.contrib.layers.fully_connected(P, num_outputs=10, activation_fn=None)
    return Zl
def compute_cost(Z, Y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z, labels=Y))
def cnn(X_train, Y_train, X_test, layers, learning_rate=0.01, minibatch_size=64, num_epochs=100, print_costs=False):
    tf.reset_default_graph()
    m, n_h, n_w, n_c = X_train.shape
    n_y = Y_train.shape[1]
    X, Y = create_placeholders(n_h, n_w, n_c, n_y)
    parameters = initialize_parameters(layers)
    Y_pred = forward_propagation(X, parameters)
    cost = compute_cost(Y_pred, Y)
    costs = []
    rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(cost)
    
    num_minibatches = m // minibatch_size
    init = tf.group(tf.initialize_all_variables(), tf.local_variables_initializer())
    
    with tf.Session() as sess:
        sess.run(init)
        for num in range(num_epochs):
            permutation = np.random.permutation(m)
            X_train, Y_train = X_train[permutation, :], Y_train[permutation, :]
            epoch_cost = 0
            for mb in range(num_minibatches):
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: X_train[mb * minibatch_size : (mb + 1) * minibatch_size, :], Y: Y_train[mb * minibatch_size : (mb + 1) * minibatch_size, :], rate: learning_rate  / (1 + 0.009 * num)})
                epoch_cost += minibatch_cost
            costs.append(epoch_cost / num_minibatches)
            
            if print_costs and num % 10 == 0:
                print('Cost after epoch', num, '=', epoch_cost / num_minibatches)
        
        plt.plot(costs[1:])
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.show()
        
        parameters = sess.run(parameters)
        correct_predictions = tf.equal(tf.argmax(Y, axis=1), tf.argmax(Y_pred, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))
        print('Accuracy:', accuracy.eval({X: X_train, Y: Y_train}))
        Y_test = sess.run(tf.argmax(Y_pred, 1), feed_dict={X: X_test})
        
    return Y_test
permutation = np.random.permutation(42000)
X_train = X_train[permutation]
Y_train = Y_train[permutation]
Y_test = cnn(X_train, Y_train, X_test, [[6, 6, 1, 8], [4, 4, 8, 15]], num_epochs=50, learning_rate=0.0005, print_costs=True)
Y_test = np.column_stack((np.arange(1, 28001), Y_test)).astype(int)
np.savetxt('submission.csv', Y_test, fmt='%i', header='ImageId,Label', comments='', delimiter=',')