import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
df = pd.read_csv('../input/Churn_Modelling.csv')
df.head()
geograpy_onehot = pd.get_dummies(df.Geography)

df = df.drop('Geography', axis=1)

df = df.join(geograpy_onehot)



gender_onehot = pd.get_dummies(df.Gender)

df = df.drop('Gender', axis=1)

df = df.join(gender_onehot)



lableenc = LabelEncoder()

df.Surname = lableenc.fit_transform(df.Surname)



df.head()
def split_train_test(dataset):

    dataset_length = len(dataset.index)

    train_length = dataset_length*0.8 - 1

    train = df.loc[:train_length,:]

    test = df.loc[train_length+1:,:]

    return train, test
train, test = split_train_test(df)
def split_fea_lab(dataset, features_col, labeles_col):

    features = dataset.loc[:, features_col]

    labeles = dataset.loc[:, [labeles_col]]

    return features, labeles
features_col = df.columns

features_col = features_col.drop(['Exited', 'RowNumber', 'CustomerId'])

labels_col = 'Exited'

features, labels = split_fea_lab(train, features_col, labels_col)

test_features, test_labels = split_fea_lab(test, features_col, labels_col)
scaler = MinMaxScaler()

features = scaler.fit_transform(features)

test_features = scaler.fit_transform(test_features)
def reshape_fea_lab(features, labels):

    features =    features.T

    labels =      labels.T

    return features, labels
features, labels = reshape_fea_lab(features, labels)

test_features, test_labels = reshape_fea_lab(test_features, test_labels)
def parameter_initialization(n_x, n_h, n_y):

    W1 = np.random.randn(n_h, n_x) * 0.01

    b1 = np.zeros((n_h,1))

    W2 = np.random.randn(n_y, n_h) * 0.01

    b2 = np.zeros((n_y,1))

    

    assert (W1.shape == (n_h, n_x))

    assert (b1.shape == (n_h,1))

    assert (W2.shape == (n_y, n_h))

    assert (b2.shape == (n_y,1))

    

    parameters = {

        'W1':W1,

        'b1':b1,

        'W2':W2,

        'b2':b2

    }

    return parameters
n_x = features.shape[0]

n_h = int(np.ceil((features.shape[0]+labels.shape[0])/2))

n_y = labels.shape[0]

parameters = parameter_initialization(n_x, n_h, n_y)
def sigmoid(x):

    return 1/(1+np.exp(-x))
def forward_propagation(features, parameters):

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    

    Z1 = np.dot(W1, features) + b1

    A1 = np.tanh(Z1)

    Z2 = np.dot(W2, A1) + b2

    A2 = sigmoid(Z2)

    

    assert (A2.shape == (1, features.shape[1]))

    

    cache = {

        'Z1':Z1,

        'A1':A1,

        'Z2':Z2,

        'A2':A2

    }

    return A2, cache
A2, cache = forward_propagation(features, parameters)
def compute_cost(A2, labels):

    m = labels.shape[1]

    logprobs =  np.dot(labels, np.log(A2.T)) + np.dot(1-labels, np.log(1-A2.T))

    cost = -(1/m) * np.sum(logprobs)

    assert(isinstance(cost, float))

    return cost
cost = compute_cost(A2, labels)

print('cost ' + str(cost))
def backward_propagation(cache, parameters, features, labels):

    W1 = parameters['W1']

    W2 = parameters['W2']

    

    A1 = cache['A1']

    A2 = cache['A2']

    

    m = features.shape[1]

    

    dZ2 = A2-labels

    dZ2 = np.array(dZ2)

    dW2 = (1/m)*np.dot(dZ2, A1.T)

    db2 = (1/m)* np.sum(dZ2, axis=1, keepdims=True)

    

    dZ1 = np.multiply(np.dot(dW2.T, dZ2), (1-np.power(A1,2)))

    dZ1 = np.array(dZ1)

    dW1 = (1/m)*np.dot(dZ1, features.T)

    db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)

    

    grads = {

        'dW1':dW1,

        'db1':db1,

        'dW2':dW2,

        'db2':db2

    }

    

    return grads
grads = backward_propagation(cache, parameters, features, labels)
def update_parameters(parameters, grads, learning_rate=0.01):

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    

    dW1 = grads['dW1']

    db1 = grads['db1']

    dW2 = grads['dW2']

    db2 = grads['db2']

    

    W1 = W1-learning_rate*dW1

    b1 = b1=learning_rate*db1

    W2 = W2-learning_rate*dW2

    b2 = b2-learning_rate*db2

    

    parameters= {

        'W1': W1,

        'b1': b1,

        'W2': W2,

        'b2': b2

    }

    return parameters
parameters = update_parameters(parameters, grads)
def ANN_Model(features, labels, num_iterations, print_cost=False):

    parameters = parameter_initialization(n_x,n_h,n_y)

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    costs = []

    for i in range(0,num_iterations):

        #forward propagation

        A2, cache = forward_propagation(features, parameters)

        

        #cost

        cost = compute_cost(A2, labels)

        

        #backward propagation

        grads = backward_propagation(cache, parameters, features, labels)

        

        #update parameters

        parameters = update_parameters(parameters, grads)

        

        #print cost

        if print_cost and i % 1000 == 0:

            print ("Cost after iteration %i: %f" %(i, cost))

        if print_cost and i % 100 == 0:

            costs.append(cost)

                

    # plot the cost

    plt.plot(costs)

    plt.ylabel('cost')

    plt.xlabel('epochs (per 100)')

    plt.title("Learning rate = 0.01")

    plt.show()

            

    return parameters
parameters = ANN_Model(features, labels, num_iterations=10000, print_cost=True)
def predict(parameters, features):

    A2, cache = forward_propagation(features, parameters)

    predictions = np.round(A2)

    

    return predictions
predictions = predict(parameters, features)
print ('Training Accuracy: %d' % float((np.dot(labels,predictions.T) + np.dot(1-labels,1-predictions.T))/float(labels.size)*100) + '%')
test_predictions = predict(parameters, test_features)
print ('Testing Accuracy: %d' % float((np.dot(test_labels,test_predictions.T) + np.dot(1-test_labels,1-test_predictions.T))/float(test_labels.size)*100) + '%')