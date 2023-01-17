import numpy as np
np.random.seed(42)
X = np.random.rand(100,1)
y = 2 + X+np.random.randn(100,1)/7.5
import matplotlib.pyplot as plt
plt.plot(X, y, 'ro')
plt.show()
def computeModelParameters(X,y):
    X_b = np.c_[np.ones((100,1)), X] # concatenate a weight of 1 to each instance
    optimal_theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return optimal_theta

theta = computeModelParameters(X,y)
theta
def predictY(x, theta): # predicts a single y value
    return theta[0]+theta[1]*x

def predictAllY(X, theta): # predicts all y values of a matrix
    X_b = np.c_[np.ones((len(X),1)), X] # concatenate 1's for theta_0 * x_0 (because x_0 doesn't exist in our data)
    y_predict = X_b.dot(theta)
    return y_predict
    
y_pred = predictAllY(X, theta)    
plt.plot(X, y, 'ro')
plt.plot(X, y_pred, '-')
plt.show()
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
print('Theta 0:', lin_reg.intercept_)
print('Theta 1:', lin_reg.coef_[0])
def calculateMSE(X, y, theta):
    sum = 0
    m = len(X)
    X_b = np.c_[np.ones((m,1)), X] # concatenate 1's for theta_0 * x_0 (because x_0 doesn't exist in our data)
    for i in range(m):
        # Create Prediction Value
        pred = theta.T.dot(X_b[i])
        # Find the Error
        error = pred - y[i]
        # Square the Error
        sqrError = error**2
        # Add the sqrError Up
        sum += sqrError
    return (1/m)*sum[0]

calculateMSE(X,y,theta)
from sklearn.metrics import mean_squared_error

mean_squared_error(y_pred=y_pred, y_true=y)