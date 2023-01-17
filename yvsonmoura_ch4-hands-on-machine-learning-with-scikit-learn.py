## Page 116

# The Normal Equation - To find the value of θ that minimizes the cost function, there is a closed-form solution.
import numpy as np
import matplotlib.pyplot as plt

import random
random.seed(99)

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

plt.style.use('ggplot')
plt.scatter(X, y, s=7)
plt.xlabel(r'$X_1$')
plt.ylabel('y')

## Page 117

# Computing θ using the Normal Equation and find best theta to fit the data
X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) # inv() - inverse of the matrix; dot() - Matrix multiplication // Inner product
print('theta_best: ', theta_best)

# Predicting possible results from the model obtained by the solution of the equation y = θ^(T).x
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)
print('X_new: ', X_new)
print('y_predict: ', y_predict)

# Plot
plt.plot(X_new, y_predict, "r-", label='Predictions')
plt.plot(X, y, "b.")
plt.xlabel(r'$X_1$')
plt.ylabel('y')
plt.axis([0, 2, 0, 15])
plt.legend()
plt.show()
## Page 118

# Performing linear regression using Scikit-Learn
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print('Intercept: ',lin_reg.intercept_, ', Coefficient: ',lin_reg.coef_)

print('Prediction of X_new: ', lin_reg.predict(X_new))

theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
print('Best Theta (Least Squares): ', theta_best_svd)

print('Best Theta (pseudoinverse of X): ', np.linalg.pinv(X_b).dot(y))

## Page 124/125

# Gradient Descent (GD is a very generic optimization algorithm capable of finding optimal solutions to a wide range of problems)

# Batch Gradient Descent
eta = 0.1 # learning rate
n_iterations = 1000
m = 100

theta = np.random.randn(2,1) # random initialization

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
theta

# Plot
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
x = np.arange(3)

# Graph 1
axes[0].scatter(X, y, s=10)
axes[0].set_title(r'$\eta = 0.02$')
axes[0].set_xlabel(r'$X_1$', fontsize=20)
axes[0].set_ylabel('y', fontsize=20)

theta = np.random.randn(2,1) # random initialization
for i in range(10):
    eta=0.02
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
    axes[0].plot(theta[0][0] + x*theta[1][0], color = [i/10,(((i*0.3))/10),0.5], lw=1, label=i)
    axes[0].legend(loc='upper left', fontsize=10, title = 'Iteration', framealpha=0.5)


# Graph 2
axes[1].scatter(X, y, s=10)
axes[1].set_title(r'$\eta = 0.1$')
axes[1].set_xlabel(r'$X_1$', fontsize=20)


theta = np.random.randn(2,1) # random initialization
for i in range(10):
    eta=0.1
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
    theta
    axes[1].plot(theta[0][0] + x*theta[1][0], color = [i/10,(((i*0.3))/10),0.5], lw=1, label=i)
    axes[1].legend(loc='upper left', fontsize=10, title = 'Iteration', framealpha=0.5)
    


# Graph 3
axes[2].scatter(X, y, s=10)
axes[2].set_title(r'$\eta = 0.5$')
axes[2].set_xlabel(r'$X_1$', fontsize=20)


theta = np.random.randn(2,1) # random initialization
for i in range(10):
    eta=0.5
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
    axes[2].plot(theta[0][0] + x*theta[1][0], color = [i/10,(((i*0.3))/10),0.5], lw=1, label=i)
    axes[2].legend(loc='upper left', fontsize=10, title = 'Iteration', framealpha=0.5)

plt.tight_layout()

## Page 127/128

# This code implements Stochastic Gradient Descent using a simple learning schedule

n_epochs = 1 # By convention we iterate by rounds of m iterations; each round is called an epoch
t0, t1 = 5, 50 # learning schedule hyperparameters
m = 20 # number of iterations

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1) # random initialization
x = np.arange(3)

fig = plt.subplots(figsize=(15,10))

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        plt.plot(theta[0][0] + x*theta[1][0], color = [i/25,(((i*0.1))/20),0.6], lw=1, label=i+1)
        

# Plot Adjustments
plt.title('Stochastic Gradient Descent - 20 first steps')
plt.scatter(X, y, s=20)
plt.xlabel(r'$X_1$', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.legend(loc='upper left', fontsize=9, title = 'Iteration', framealpha=0.5)


## Page 128

# To perform Linear Regression using SGD with Scikit-Learn - SGDRegressor
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())

sgd_reg.intercept_, sgd_reg.coef_
## Page 129

# Stochastic / Batch / Mini-batch -> Gradient Descent

n_epochs = 1 # By convention we iterate by rounds of m iterations; each round is called an epoch
t0, t1 = 5, 50 # learning schedule hyperparameters
m = 100 # number of iterations

def learning_schedule(t):
    return t0 / (t + t1)

random.seed(42)
theta = np.random.randn(2,1) # random initialization

fig = plt.subplots(figsize=(14,8))

# Stochastic Gradient Descent
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        stoch = plt.plot(theta[0][0], theta[1][0], color = [(i*0.5)/100,(((i*0.95))/100),0.1], marker='x', label='Stochastic')
        

# Batch Gradient Descent
theta = np.random.randn(2,1) # random initialization
for i in range(100):
    eta=0.08
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
    batch = plt.plot(theta[0][0], theta[1][0], color = [(i*0.1)/100,(((i*0.5))/100),0.8], marker='o', label='Batch')
    

# Mini-batch Gradient Descent
n_epochs = 1 # By convention we iterate by rounds of m iterations; each round is called an epoch
t0, t1 = 5, 50 # learning schedule hyperparameters
m = 100 # number of iterations
batch_size = 5
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+batch_size]
        yi = y[random_index:random_index+batch_size]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        mini = plt.plot(theta[0][0], theta[1][0], color = [i/100,(((i*0.01))/100),0.8], marker='s', label='Mini-batch')
  

# Plot Adjustments
plt.title('Gradient Descent Methods')
plt.xlabel(r'$\theta_0$', fontsize=20)
plt.ylabel(r'$\theta_1$', fontsize=20)
plt.xlim(0, 7)
plt.ylim(0, 7)
plt.legend(handles=[batch[0],stoch[0],mini[0]], loc='upper left', fontsize=12, framealpha=0.5, title='Last Iteration', title_fontsize=13)
## Page 130

# Polynomial Regression
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)


from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_

a = lin_reg.coef_[0][1]
b = lin_reg.coef_[0][0]
c = lin_reg.intercept_[0]
XX = np.arange(-3, 3, 0.1)


fig = plt.subplots(1, 1, figsize=(15,8))
plt.scatter(X, y, color='b', label='Data')
plt.plot(XX, a*(XX**2)+b*(XX)+c, linestyle='-', lw=2, label='Prediction')
plt.xlabel(r'$X_1$', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.legend(fontsize=15)

## Page 133

# Learning Curve

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    fig = plt.subplots(1,1, figsize=(15, 8))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=1.5, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=1.5, label="val")
    plt.xlabel('Training set size', fontsize=20, color='k')
    plt.ylabel('RMSE', fontsize=20, color='k')
    plt.legend(fontsize=15)
    
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
## Page 135

# Learning curves for the polynomial model
from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),])

plot_learning_curves(polynomial_regression, X, y)
plt.ylim(0, 4)
plt.title('Polynomial Model', fontsize=18)
## Page 138/139

# Ridge Regression

from sklearn.linear_model import Ridge

# Ridge Regression with Scikit-Learn using a closed-form solution
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])


# Stochastic Gradient Descent
sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])
## Page 141

# Lasso Regression
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])
## Page 142

# Elastic Net
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])
## Page 143a

# Early Stopping
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.model_selection import train_test_split

random.seed(99)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20)

# Prepare the data
poly_scaler = Pipeline([
    ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
    ("std_scaler", StandardScaler())
    ])

X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)
sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True, penalty=None, learning_rate="constant", eta0=0.0003) # learning rate different from the original one

minimum_val_error = float("inf")
best_epoch = None
best_model = None
train_error, val_error = [], []

for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train.ravel(order='C')) # continues where it left off
    
    y_train_predict = sgd_reg.predict(X_train_poly_scaled)
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    
    train_error.append(mean_squared_error(y_train.ravel(), y_train_predict))
    val_error.append(mean_squared_error(y_val, y_val_predict))
    
    if val_error[epoch] < minimum_val_error:
        minimum_val_error = val_error[epoch]
        best_epoch = epoch
        best_model = clone(sgd_reg)

## Page 143b

# Plot - Early stopping regularization
fig = plt.subplots(1,1, figsize=(16, 9))
plt.plot(np.arange(1000), train_error, lw=2, color='r', linestyle='-', label='Training Set')
plt.plot(np.arange(1000), val_error, lw=2, color='b', label='Validation Set')
plt.hlines(y=minimum_val_error, xmin=0, xmax=1000, lw=2, colors='k', linestyles='dashed', label='Best Model')
plt.ylim(0, 8)
plt.xlabel('Epoch', fontsize=15, color='k')
plt.ylabel('RMSE', fontsize=15, color='k')
plt.legend(fontsize=15)
## Page 144

# Logistic Regression
t = np.arange(-10,10, step=0.1)
log_function = 1/(1+(np.exp(-t)))

fig = plt.subplots(1,1, figsize=(12, 6))
plt.plot(t, log_function, color='b', label=r'$\sigma (t) = \frac{1}{1+e^{-t}}$')
plt.hlines(0.5, xmin=-10, xmax=10, linestyle='dashed', color='k')
plt.hlines(1, xmin=-10, xmax=10, linestyle='dashed', color='k')
plt.vlines(0, ymin=0, ymax=1, color='k')
plt.xlabel('t', color='k', fontsize=15)
plt.legend(fontsize=20, loc='lower right')
## Page 147

# Decision Boundary
from sklearn import datasets

iris = datasets.load_iris()
list(iris.keys())

X = iris["data"][:, 3:] # petal width
y = (iris["target"] == 2).astype(np.int) # 1 if Iris-Virginica, else 0

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)

fig = plt.subplots(1, 1, figsize=(14, 6))
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
plt.vlines(X_new[np.argmax(y_proba[:,1]>=0.5)], ymin=0, ymax=1, color='k', linestyle='dashed', label='Decision Boundary')
plt.scatter(X[np.where(y==1)], y[np.where(y==1)], marker='^', s=100, color='g') # Iris-Virginica
plt.scatter(X[np.where(y==0)], y[np.where(y==0)], marker='s', s=100, color='b') # Not Iris-Virginica
plt.xlabel('Petal width (cm)', color='k')
plt.ylabel('Probability', color='k')
plt.legend(loc='center left', fontsize=10)

## Page 149

# Softmax Regression - it supports multiple classes directly without having to train and combine multiple binary classifiers
X = iris["data"][:, (2, 3)] # petal length, petal width
y = iris["target"]
softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)
softmax_reg.fit(X, y)

print(softmax_reg.predict([[5, 2]])) # Class 2 = Iris-Virginica
print(softmax_reg.predict_proba([[5, 2]])) # Probabilities for being class 0, class 1 and class 2.
