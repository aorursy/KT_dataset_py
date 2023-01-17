import numpy as np

import pandas as pd

from sklearn.datasets import load_boston

import plotly.express as px
# Load the data and convert it to a DataFrame

data = load_boston()

df = pd.DataFrame(data['data'],columns=data['feature_names'])

df.insert(13,'Y',data['target'])

df.head(5)
X = df[df.columns[:-1]]

Y = df['Y']
X_norm = (X - X.min()) / (X.max() - X.min())

X = X_norm
theta = np.zeros(X.shape[1])
def cost_function(X,Y,theta):

    m = len(Y)

    prediction = np.dot(X,theta.T)

    return (1/(m)) * np.sum((prediction - Y) ** 2)
# To test that our cost function is working correctly

from sklearn.metrics import mean_squared_error
cost_function(X,Y,theta)
mean_squared_error(np.dot(X,theta.T),Y)
def batch_gradient_descent(X,Y,theta,alpha,iters):

    cost_history = [0] * iters

    #initalize our cost history list to store the cost function on every iteration

    for i in range(iters):

        prediction = np.dot(X,theta.T)

        

        theta = theta - (alpha/len(Y)) * np.dot(prediction - Y,X)

        cost_history[i] = cost_function(X,Y,theta)

    return theta,cost_history
%%time

batch_theta,batch_history = batch_gradient_descent(X,Y,theta,0.05,5000)
cost_function(X,Y,batch_theta)
fig = px.line(batch_history,x=range(5000),y=batch_history,labels={'x':'no. of iterations','y':'cost function'})

fig.show()
def mini_batch_gradient_descent(X,Y,theta,alpha,iters,batch_size=10):

    for i in range(iters):

        cost_history = [0] * iters

        for j in range(0,len(Y),batch_size):

            theta = theta - (alpha/batch_size) * (np.dot(np.dot(X,theta.T) - Y,X))

            cost_history[i] = cost_function(X,Y,theta)

        return theta,cost_history
%%time

mini_batch_theta,mini_batch_history = mini_batch_gradient_descent(X,Y,theta,0.05,3000)
cost_function(X,Y,mini_batch_theta)