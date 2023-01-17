import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



train_data = pd.read_csv('..//input//train.csv')

test_data = pd.read_csv('..//input//test.csv')



train_label = train_data['label']

train_data = train_data.drop('label', axis = 1)

#print (train_pixel)
import math

import random



f, ax = plt.subplots(5,5)

[rown, coln] = train_data.shape

for i in range(1,26):

    j = random.randrange(0, coln-1)

    display_data = train_data.iloc[j,:].values

    nrows, ncols = 28, 28

    image = display_data.reshape((nrows, ncols))

    n = math.ceil(i/5)-1

    m = [0,1,2,3,4]*5

    img = ax[m[i-1],n].imshow(image)

    img.set_cmap('hot')

    

plt.show()
def one_hot_vectors(y):

    n_val = np.max(y) + 1

    return (np.eye(n_val, dtype='int')[y])

#print (y)
def sigmoid(z):

    g_z = np.divide(1.0, (1.0 + np.exp(-np.array(z))))

    return g_z

#print (sigmoid(([0,1,2,3])))
def weightsRand(L_out, L_in, epsilon_init = 0.12):

    return np.random.uniform(size = (L_out, L_in)) * 2 * epsilon_init - epsilon_init
def forward_propogation(X, theta1, theta2):

    [m, n] = X.shape

    a1 = np.c_[np.ones(m, int), X]

    z2 = np.matmul(theta1, np.transpose(a1))

    a2 = sigmoid(z2)

    [m, n] = a2.shape

    a2 = np.c_[np.ones(n,int), np.transpose(a2)]

    z3 = np.matmul(theta2, np.transpose(a2))

    a3 = sigmoid(z3)

    return (a1, a2, a3, z2)



#a3 = forward_propogation(train_data.values, weightsRand(25, 785), weightsRand(10, 26))
def sigmoidGradient(z):

    #print (np.dot(sigmoid(z), (1-sigmoid(z))).shape)

    return np.multiply(sigmoid(z), (1-sigmoid(z)))
def backpropogation(y, a1, a2, a3, z2, theta1, theta2, _lambda):

    err3 = a3 - np.transpose(y) #42000x10

    [m,n] = z2.shape

    err2 = np.multiply(np.dot(err3,theta2), np.c_[np.ones(n, int), np.transpose(sigmoidGradient(z2))])[:,1:]

    

    theta2_grad = ((1/m) * np.dot(np.transpose(err3),a2)) + ((_lambda/m) * theta2)

    theta1_grad = ((1/m) * np.dot(np.transpose(err2),a1)) + ((_lambda/m) * theta1)

    

    return theta1_grad, theta2_grad
def cost_function(theta, X, y, _lambda):

    theta1 = np.reshape(theta[:(28*785)],[28, 785])

    theta2 = np.reshape(theta[(28*785):], [10, 29])

    a1, a2, a3, z2 = forward_propogation(X, theta1, theta2)

    [m, n] = a3.shape

    y = np.transpose(y)

    a3 = np.transpose(a3)

    J = np.sum(np.sum((np.dot(-y,(np.log(a3))))-(np.dot((1-y),(np.log(1-a3))))))

    J += (_lambda/(2*m)) + np.sum(np.sum(theta1[:, 1:]**2)) + np.sum(np.sum(theta2[:, 1:]**2))

    theta1_grad, theta2_grad = backpropogation(y, a1, a2, a3, z2, theta1, theta2, _lambda)

    #return (J, theta1_grad, theta2_grad)

    print (np.append(theta1_grad.ravel(), theta2_grad.ravel()).shape)

    return (np.append(theta1_grad.ravel(), theta2_grad.ravel()))



#cost_function(a3, y, weightsRand(25, 785), weightsRand(10, 26), 1)
input_layer_size = 784

hidden_layer_size = 28

num_labels = 10

X = train_data.values

y = one_hot_vectors(train_label.values)

theta1 = weightsRand(hidden_layer_size, input_layer_size+1)

theta2 = weightsRand(num_labels, hidden_layer_size+1)

_lambda = 1
from scipy import optimize



theta = np.append(theta1.ravel(), theta2.ravel())

#print (theta.shape)

#print (cost_function(theta, X, y, _lambda).shape)



#[res] = optimize.fmin_cg(cost_function(theta, X, y, _lambda), x0=theta, args=(X, y, _lambda), maxiter=50)

#print (res)

res = optimize.minimize(cost_function, x0=theta, method='CG',

         options = {'maxiter' :50, 'disp':True}, jac =True, args=(X, y, _lambda));
print (res)

theta1 = np.reshape(res.x[:(28*785)],[28, 785])

theta2 = np.reshape(res.x[(28*785):], [10, 29])
_, _, a3, _ = forward_propogation(test_data.values, theta1, theta2)

[m, n] = a3.shape

output_y = np.argmax(a3, axis=0)

print (output_y.shape)



f, ax = plt.subplots(1,1)

[rown, coln] = test_data.shape

#for i in range(1,2):

display_data = train_data.iloc[0,:].values

nrows, ncols = 28, 28

image = display_data.reshape((nrows, ncols))

img = ax.imshow(image)

img.set_cmap('hot')

print (output_y[0])

plt.show()
df = pd.DataFrame(data={'ImageId':range(1,n+1), 'Label':output_y}, index=None)

print (df)
df.to_csv(path_or_buf = 'output.csv', index=None, header=True)