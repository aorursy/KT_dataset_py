import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
# Load and convert data to DataFrame
data = load_boston()

df = pd.DataFrame(data['data'],columns=data['feature_names'])
df.insert(13,'target',data['target'])
df.head(5)
X,y = df.drop('target',axis=1),df['target']

thetas = np.zeros(X.shape[1])
def cost_function(X,Y,B):
    predictions = np.dot(X,B.T)
    
    cost = (1/len(Y)) * np.sum((predictions - Y) ** 2)
    return cost
cost_function(X,y,thetas)
mean_squared_error(np.dot(X,thetas.T),y)
X_norm = (X - X.min()) / (X.max() - X.min())
X = X_norm
t0,t1 = 5,50 # learning schedule hyperparams
def learning_schedule(t):
    return t0/(t+t1)
def stochastic_gradient_descent(X,y,theta,n_epochs=50):
    c_hist = [0] * n_epochs
    for epoch in range(n_epochs):
        for i in range(len(y)):
            rand_index = np.random.randint(len(y))
            ind_x = X[rand_index:rand_index+1]
            ind_y = y[rand_index:rand_index+1]

            gradients = 2 * ind_x.T.dot(ind_x.dot(theta) - ind_y)
            eta = learning_schedule(epoch * len(y) + i)
            theta = theta - eta * gradients
            c_hist[epoch] = cost_function(ind_x,ind_y,theta)
    return theta,c_hist
th_n,cost_history = stochastic_gradient_descent(X,y,thetas)
mean_squared_error(np.dot(X,th_n.T),y)
import matplotlib.pyplot as plt
plt.plot(range(50),cost_history)
plt.show()
