import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

%matplotlib inline
train = pd.read_csv('../input/train.csv')
train.drop(['PassengerId', 'Name', 'Ticket', 'Embarked'], axis=1, inplace=True)
train.describe(include='all')
train.fillna(value={'Age': train['Age'].mean()}, inplace=True)
train.dropna(axis=1, inplace=True)
train['Age'] = train['Age'] / (train['Age'].max() - train['Age'].min())
train['Fare'] = train['Fare'] / (train['Fare'].max() - train['Fare'].min())
train.describe(include='all')
X_train = np.zeros((9, train.shape[0]))
Y_train = np.zeros((train.shape[0], 1))

for i, x in train.iterrows():
    X_train[x['Pclass'] - 1, i] = 1
    if x['Sex'] == 'M':
        X_train[3, i] = 1
    else:
        X_train[4, i] = 1
    X_train[5:, i] = x['Age'], x['SibSp'], x['Parch'], x['Fare']
    Y_train[i] = x['Survived']
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def relu(z):
    return np.maximum(0, z)
def initialize_parameters(layers_dims):
    L = len(layers_dims)
    parameters = {}
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    
    return parameters
def forward_propagation(X, parameters):
    L = len(parameters) // 2
    A_prev = X
    caches = []
    
    for l in range(1, L):
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        
        Z = np.dot(W, A_prev) + b
        caches.append((A_prev, W, b, Z))
        A_prev = relu(Z)
    
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    
    Z = np.dot(W, A_prev) + b
    caches.append((A_prev, W, b, Z))
    AL = sigmoid(Z)
    
    return AL, caches
def compute_cost(AL, Y):
    m = Y.shape[0]
    
    return (1 / m) * (np.dot(np.log(AL), Y) + np.dot(np.log(1 - AL), 1 - Y))
def dsigmoid(z):
    g = sigmoid(z)
    return g * (1 - g)
def drelu(z):
    return (z > 0) * 1
def backward_propagation(AL, Y, caches):
    grads = {}
    m = Y.shape[0]
    L = len(caches)
    dAL = np.divide(1 - Y.T, 1 - AL) - np.divide(Y.T, AL)
    A_prev, W, b, Z = caches[L - 1]
    
    dZ = dAL * dsigmoid(Z)
    grads['dW' + str(L)] = (1 / m) * np.dot(dZ, A_prev.T)
    grads['db' + str(L)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    for l in reversed(range(1, L)):
        A_prev, W, b, Z = caches[l - 1]
        dZ = dA_prev * drelu(Z)
        grads['dW' + str(l)] = (1 / m) * np.dot(dZ, A_prev.T)
        grads['db' + str(l)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
    
    return grads
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    
    for l in range(1, L):
        parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]
    
    return parameters
def neural_network(X, Y, layers_dims, learning_rate=0.01, num_iterations=2500, print_costs=False):
    parameters = initialize_parameters(layers_dims)
    costs = []
    
    for i in range(num_iterations):
        AL, caches = forward_propagation(X, parameters)
        cost = compute_cost(AL, Y)
        grads = backward_propagation(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        costs.append(cost)
        if print_costs and i % 1000 == 0:
            print('Cost after ', i, ' iteration:', cost)
    
    Y_pred = (AL > 0.5) * 1
    print('Training accuracy:', (100 / Y.shape[0]) * (np.dot(Y_pred, Y) + np.dot(1 - Y_pred, 1 - Y)))
    
    plt.plot(np.squeeze(costs))
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    
    return parameters
layers_dims = [9, 50, 10, 1]
model = neural_network(X_train, Y_train, layers_dims, 5, 20000, print_costs=True)
test = pd.read_csv('../input/test.csv')
test.drop(['Name', 'Ticket', 'Embarked'], axis=1, inplace=True)
test.fillna(value={'Age': train['Age'].mean()}, inplace=True)
test.fillna(value={'Fare': train['Fare'].mean()}, inplace=True)
test.dropna(axis=1, inplace=True)
test['Age'] = test['Age'] / (test['Age'].max() - test['Age'].min())
test['Fare'] = test['Fare'] / (test['Fare'].max() - test['Fare'].min())
test.describe(include='all')
X = np.zeros((9, test.shape[0]))

for i, x in test.iterrows():
    X[x['Pclass'] - 1, i] = 1
    if x['Sex'] == 'M':
        X[3, i] = 1
    else:
        X[4, i] = 1
    X[5:, i] = x['Age'], x['SibSp'], x['Parch'], x['Fare']
def predict(X, parameters):
    L = len(parameters) // 2
    A_prev = X
    
    for l in range(1, L):
        Z = np.dot(parameters['W' + str(l)], A_prev) + parameters['b' + str(l)]
        A_prev = relu(Z)
    
    Z = np.dot(parameters['W' + str(L)], A_prev) + parameters['b' + str(L)]
    return (sigmoid(Z) > 0.5) * 1
Y_pred = predict(X, model).T
test['Survived'] = Y_pred
test[['PassengerId', 'Survived']].astype(int).to_csv('submission_neural_network.csv', index=False)
