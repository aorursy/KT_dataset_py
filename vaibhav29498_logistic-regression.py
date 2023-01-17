import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

%matplotlib inline
train = pd.read_csv('../input/train.csv')
print('The variables present in the data are:')
for x in list(train):
    print(x, end=' ')
train.drop(['PassengerId', 'Name', 'Ticket', 'Embarked'], axis=1, inplace=True)
train.describe(include='all')
train.isnull().sum()
train.fillna(value={'Age': train['Age'].mean()}, inplace=True)
train.dropna(axis=1, inplace=True)
train['Age'] = train['Age'] / (train['Age'].max() - train['Age'].min())
train['Fare'] = train['Fare'] / (train['Fare'].max() - train['Fare'].min())
train.describe(include='all')
X_train = np.zeros((train.shape[0], 9))
Y_train = np.zeros((train.shape[0], 1))

for i, x in train.iterrows():
    X_train[i, x['Pclass'] - 1] = 1
    if x['Sex'] == 'M':
        X_train[i, 3] = 1
    else:
        X_train[i, 4] = 1
    X_train[i, 5:] = x['Age'], x['SibSp'], x['Parch'], x['Fare']
    Y_train[i] = x['Survived']
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def initialize(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b
def propagate(w, b, X, Y):
    m = X.shape[0]
    A = sigmoid(np.dot(X, w) + b)
    J = (-1 / m) * (np.dot(Y.T, np.log(A)) + np.dot(1 - Y.T, np.log(1- A)))
    
    dw = (1 / m) * np.dot(X.T, A - Y)
    db = (1 / m) * np.sum(A - Y, axis=0)
    
    grads = {
        'dw': dw,
        'db': db
    }
    
    return grads, np.squeeze(J)
def optimize(w, b, X, Y, iterations, alpha, print_costs=False):
    costs = []
    for i in range(iterations):
        grads, cost = propagate(w, b, X, Y)
        costs.append(cost)
        if print_costs and (i + 1) % 100 == 0:
            print('Cost after ', i + 1, ' iterations:', cost)
        w -= alpha * grads['dw']
        b -= alpha * grads['db']
    
    params = {
        'w': w,
        'b': b
    }
    
    return params, grads, costs
def predict(w, b, X):
    m = X.shape[0]
    A = sigmoid(np.dot(X, w) + b)
    Y_prediction = np.zeros((m, 1))
    
    for i in range(m):
        if A[i] >= 0.5:
            Y_prediction[i] = 1
    
    return Y_prediction
def logistic_regression(X_train, Y_train, X, iterations, alpha, print_costs=False):
    n_train = X_train.shape[1]
    w, b = initialize(n_train)
    params, grads, costs = optimize(w, b, X_train, Y_train, iterations, alpha, print_costs)
    
    w = params['w']
    b = params['b']
    
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction = predict(w, b, X)
    
    train_accuracy = 100 - np.mean(np.abs(Y_train - Y_prediction_train), axis=0) * 100
    print('Training accuracy:', train_accuracy)
    
    model = {
        'Y_prediction_train': Y_prediction_train,
        'Y_prediction': Y_prediction,
        'w': w,
        'b': b,
        'costs': costs
    }
    
    return model
test = pd.read_csv('../input/test.csv')
test.drop(['Name', 'Ticket', 'Embarked'], axis=1, inplace=True)
test.fillna(value={'Age': train['Age'].mean()}, inplace=True)
test.fillna(value={'Fare': train['Fare'].mean()}, inplace=True)
test.dropna(axis=1, inplace=True)
test['Age'] = test['Age'] / (test['Age'].max() - test['Age'].min())
test['Fare'] = test['Fare'] / (test['Fare'].max() - test['Fare'].min())
test.describe(include='all')
X = np.zeros((test.shape[0], 9))

for i, x in test.iterrows():
    X[i, x['Pclass'] - 1] = 1
    if x['Sex'] == 'M':
        X[i, 3] = 1
    else:
        X[i, 4] = 1
    X[i, 5:] = x['Age'], x['SibSp'], x['Parch'], x['Fare']
model = logistic_regression(X_train, Y_train, X, iterations=2000, alpha=0.01, print_costs=True)
test['Survived'] = predict(model['w'], model['b'], X)
test[['PassengerId', 'Survived']].astype(int).to_csv('submission.csv', index=False)
costs = np.squeeze(model['costs'])
plt.plot(costs)
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()