%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

from datetime import datetime, date, time

from sklearn.metrics import r2_score
# data pre-processing

data = pd.read_csv('../input/kc_house_data.csv')

data['date'] = pd.to_datetime(data['date'])

data['date'] = data['date'].dt.dayofyear



# convert yr_built to age in years

currYear = datetime.now().year

data['yr_built'] = currYear - data['yr_built']

data = data.rename(columns = {'yr_built':'house_age'})



# drop id column

data = data.drop('id',axis=1)

# drop zipcode column (location already in lat/lon columns)

data = data.drop('zipcode', axis=1)



# convert yr_renovated to years since renovation

data['yr_renovated'] = currYear - data['yr_renovated']

data['yr_renovated'] = data['yr_renovated'].where(data['yr_renovated'] != currYear, 0)

data = data.rename(columns = {'yr_renovated': 'yrs_since_renov'})

data.head()
train,test = np.split(data.sample(frac=1), [int(.7*len(data))])
# convert to a matrix to allow for matrix operations on the data

train_matrix = train.as_matrix()

test_matrix = test.as_matrix()
# y is the house price

y = np.array(([train_matrix[:,1]]),dtype=float)

train_matrix = np.delete(train_matrix, 1, 1)
# X is the rest of the columns

X = np.array([train_matrix[:,:]])

l,m,n = X.shape

X.shape = (m,n)
# resizing to vectors

y_test = np.array(([test_matrix[:,1]]),dtype=float)

test_matrix = np.delete(test_matrix, 1, 1)

X_test = np.array([test_matrix[:,:]])

l_test,m_test,n_test = X_test.shape

X_test.shape = (m_test, n_test)



m_test = y_test.size

y_test = y_test.reshape(m_test,1)

y_test = np.matrix(y_test)
# setup

num_iters = 500;

alpha = 0.01;

theta = np.zeros((n+1,1),dtype=np.int)

theta = np.matrix(theta)

lam = 1
def featureNormalize(A):

    mu = np.mean(A, axis=0)

    m = mu.size

    mu = mu.reshape(1,m)

    sigma = np.std(A, axis=0)

    sigma = sigma.reshape(1,m)

    A_norm = (A - mu) / sigma

    return A_norm
# normalize x for training and test set. No need to normalize y. 

X_norm = featureNormalize(X)

X_test_norm = featureNormalize(X_test)
# add a columns of ones for the bias unit to the training set

m,n = X.shape

I = np.ones((m,1),dtype=int)

X_norm = np.c_[I,X_norm]
# add a columns of ones for the bias unit to the test set

m_test,n_test = X_test.shape

I = np.ones((m_test,1),dtype=int)

X_test_norm = np.c_[I,X_test_norm]
def computeCost(X,y,theta, lam):

    m = y.size

    h = X*theta

    n = theta.size

    theta_tmp = np.matrix(theta)

    theta_tmp[[1,1]] = 0 # this will make sure theta_0 is not updated

    J = 1./(2.*m) * np.sum(np.square(h-y)) + (lam/(2.*m))* np.sum(np.square(theta_tmp))

    return J
# some resizing, test cost function

m = y.size

y = y.reshape(m,1)

y = np.matrix(y)

computeCost(X_norm,y,theta,lam)
def gradientDescent(X, y, theta, alpha, num_iters, lam):

    m = y.size

    J_history = np.zeros((num_iters,1),dtype=float)

    J_history = np.matrix(J_history)

    

    for i in range(1,num_iters):

        h = X * theta

        theta_tmp = np.matrix(theta)

        theta_tmp[[1,1]] = 0 # this will make sure theta_0 is not updated

    

        delta = (1./m) * (X.T * (h-y)) + np.sum((lam/m)*theta_tmp)

        theta = theta - alpha*delta  

        J_history[[i,1]] = computeCost(X,y,theta,lam)

    return J_history,theta
J_history,theta = gradientDescent(X_norm, y, theta, alpha, num_iters, lam)
# plot cost function

fig, ax = plt.subplots()

x_axis = np.arange(1,num_iters+1)

x_axis = x_axis.reshape(num_iters,1)

ax.plot(x_axis, J_history, label=".01")

legend = ax.legend(loc='upper center', shadow=True)

plt.ylabel("Cost (J)")

plt.xlabel("Num Iterations")

plt.show()
# predictions and r-square

y_pred = X_test_norm * theta;

m_pred = y_pred.size

y_pred = y_pred.reshape(m_pred,1)

y_pred = np.matrix(y_pred)

r2 = r2_score(y_test, y_pred)

print (r2)