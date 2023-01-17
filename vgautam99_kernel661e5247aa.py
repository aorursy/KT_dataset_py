import math

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
df = pd.read_csv("../input/Churn_Modelling.csv")
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
def random_mini_batch(features, labels, mini_batch_size=64, seed=0):

    m = features.shape[1]

    mini_batches = []

    

    np.random.seed(seed)

    

    permutation = list(np.random.permutation(m))

    shuffled_features = features[:, permutation]

    shuffled_labels = labels.iloc[:, permutation]

    

    num_complete_mini_batches = math.floor(m/mini_batch_size)

    

    for k in range(num_complete_mini_batches):

        mini_batch_features = shuffled_features[:, k*mini_batch_size:(k+1)*mini_batch_size]

        mini_batch_lables = shuffled_labels.iloc[:, k*mini_batch_size:(k+1)*mini_batch_size]

        mini_batch = (mini_batch_features, mini_batch_lables)

        mini_batches.append(mini_batch)

        

    if m%mini_batch_size !=0:

        end = m - mini_batch_size * math.floor(m/mini_batch_size)

        mini_batch_features = shuffled_features[:, mini_batch_size*num_complete_mini_batches:]

        mini_batch_lables = shuffled_labels.iloc[:, mini_batch_size*num_complete_mini_batches:]

        mini_batch = (mini_batch_features, mini_batch_lables)

        mini_batches.append(mini_batch)

        

    return mini_batches
mini_batches = random_mini_batch(features, labels)

print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))

print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))

print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))

print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))

print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape)) 

print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))

print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))
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
def initialize_velocity(parameters):

    L = len(parameters) // 2

    v = {}

    

    for l in range(L):

        v['dW'+str(l+1)] = np.zeros_like(parameters['W'+str(l+1)])

        v['db'+str(l+1)] = np.zeros_like(parameters['b'+str(l+1)])

        

    return v
v = initialize_velocity(parameters)

print("v[\"dW1\"] = " + str(v["dW1"]))

print("v[\"db1\"] = " + str(v["db1"]))

print("v[\"dW2\"] = " + str(v["dW2"]))

print("v[\"db2\"] = " + str(v["db2"]))
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
def update_parameters_with_momentum(parameters, grads,v, beta, learning_rate=0.01):

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    

    dW1 = grads['dW1']

    db1 = grads['db1']

    dW2 = grads['dW2']

    db2 = grads['db2']

    

    v['dW1'] = beta*v['dW1'] + (1-beta)*dW1

    v['db1'] = beta*v['db1'] + (1-beta)*db1

    v['dW2'] = beta*v['dW2'] + (1-beta)*dW2

    v['db2'] = beta*v['db2'] + (1-beta)*db2

    

    W1 = W1-learning_rate*v['dW1']

    b1 = b1=learning_rate*v['db1']

    W2 = W2-learning_rate*v['dW2']

    b2 = b2-learning_rate*v['db2']

    

    parameters= {

        'W1': W1,

        'b1': b1,

        'W2': W2,

        'b2': b2

    }

    return parameters, v
parameters, v = update_parameters_with_momentum(parameters, grads, v, beta=0.9)
def ANN_Model(features, labels, num_iterations, print_cost=False):

    parameters = parameter_initialization(n_x,n_h,n_y)

    v = initialize_velocity(parameters)

    seed = 10

    costs = []

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    

    for i in range(0,num_iterations):

        seed = seed+1

        mini_batches = random_mini_batch(features=features, labels=labels, seed=seed)

        

        for mini_batch in mini_batches:

            #mini batch

            (mini_batch_features, mini_batch_lables) = mini_batch

            

            #forward propagation

            A2, cache = forward_propagation(mini_batch_features, parameters)



            #cost

            cost = compute_cost(A2, mini_batch_lables)



            #backward propagation

            grads = backward_propagation(cache, parameters, mini_batch_features, mini_batch_lables)



            #update parameters

            parameters, v = update_parameters_with_momentum(parameters, grads, v, beta=0.9)



        #print cost

        if print_cost and i % 1000 == 0:

            print ("Cost after iteration %i: %f" %(i, cost))

        if print_cost and i % 100 == 0:

            costs.append(cost)

    

    plt.plot(costs)

    plt.ylabel('cost')

    plt.xlabel('epochs (per 100)')

    plt.title("Learning rate = 0.01")

    plt.show()

            

    return parameters
parameters = ANN_Model(features, labels, num_iterations=9000, print_cost=True)
def predict(parameters, features):

    A2, cache = forward_propagation(features, parameters)

    predictions = np.round(A2)

    

    return predictions
predictions = predict(parameters, features)
print ('Training Accuracy: %d' % float((np.dot(labels,predictions.T) + np.dot(1-labels,1-predictions.T))/float(labels.size)*100) + '%')
test_predictions = predict(parameters, test_features)
print ('Testing Accuracy: %d' % float((np.dot(test_labels,test_predictions.T) + np.dot(1-test_labels,1-test_predictions.T))/float(test_labels.size)*100) + '%')