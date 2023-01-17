import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
digits = load_digits()
digits.data.shape
import matplotlib.pyplot as plt
shape=(np.reshape(digits.data[1],(8,8)))
plt.imshow(shape)
plt.show()
digits.target.shape
X = digits['data'][(digits['target']==0) | (digits['target']==1)]
y = np.array([])
for i in range(0,digits['target'].shape[0]):
    if((digits['target'][i]==0) or (digits['target'][i]==1)):
        y = np.append(y,[digits['target'][i]],axis=0)
y = np.reshape(y,(y.shape[0],1))
print(X.shape)
print(y.shape)
def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))
def predict(features, weights):
  '''
  Returns 1D array of probabilities
  that the class label == 1
  '''
  z = np.dot(features, weights)
  return sigmoid(z)
def cost_function(features, labels, weights):
    '''
    Using Mean Absolute Error

    Features:(100,3)
    Labels: (100,1)
    Weights:(3,1)
    Returns 1D matrix of predictions
    Cost = ( log(predictions) + (1-labels)*log(1-predictions) ) / len(labels)
    '''
    observations = len(labels)

    predictions = predict(features, weights)

    #Take the error when label=1
    class1_cost = -labels*np.log(predictions)

    #Take the error when label=0
    class2_cost = (1-labels)*np.log(1-predictions)

    #Take the sum of both costs
    cost = class1_cost - class2_cost

    #Take the average cost
    cost = cost.sum()/observations

    return cost
def update_weights(features, labels, weights, lr):
    '''
    Vectorized Gradient Descent

    Features:(200, 3)
    Labels: (200, 1)
    Weights:(3, 1)
    '''
    N = len(features)

    #1 - Get Predictions
    predictions = predict(features, weights)

    #2 Transpose features from (200, 3) to (3, 200)
    # So we can multiply w the (200,1)  cost matrix.
    # Returns a (3,1) matrix holding 3 partial derivatives --
    # one for each feature -- representing the aggregate
    # slope of the cost function across all observations
    gradient = np.dot(features.T,  predictions - labels)

    #3 Take the average cost derivative for each feature
    gradient /= N

    #4 - Multiply the gradient by our learning rate
    gradient *= lr

    #5 - Subtract from our weights to minimize cost
    weights -= gradient

    return weights
def train(features, labels, weights, lr, iters):
    cost_history = []

    for i in range(iters):
        weights = update_weights(features, labels, weights, lr)

        #Calculate error for auditing purposes
        cost = cost_function(features, labels, weights)
        cost_history.append(cost)

        # Log Progress
        if i % 1000 == 0:
            print ("iter: "+str(i) + " cost: "+str(cost))

    return weights, cost_history
def accuracy(predicted_labels, actual_labels):
    diff = predicted_labels - actual_labels
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))
features = X
labels = y
weights = np.zeros(features.shape[1])
weights = np.reshape(weights,(features.shape[1],1))
lr = 0.001
iters = 1001

print(features.shape)
print(labels.shape)
print(weights.shape)
train(features,labels,weights,lr,iters)
probabilities = predict(features, weights).flatten()
# classifications = classify(probabilities)
# our_acc = accuracy(classifications,labels.flatten())
# print('Our score: ',our_acc)
print(np.round(probabilities))

