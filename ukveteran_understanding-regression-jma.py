%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt



seed = 1            # seed for random number generation 

numInstances = 200  # number of data instances

np.random.seed(seed)

X = np.random.rand(numInstances,1).reshape(-1,1)

y_true = -3*X + 1 

y = y_true + np.random.normal(size=numInstances).reshape(-1,1)



plt.scatter(X, y,  color='black')

plt.plot(X, y_true, color='blue', linewidth=3)

plt.title('True function: y = -3X + 1')

plt.xlabel('X')

plt.ylabel('y')
numTrain = 20   # number of training instances

numTest = numInstances - numTrain



X_train = X[:-numTest]

X_test = X[-numTest:]

y_train = y[:-numTest]

y_test = y[-numTest:]
from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score



# Create linear regression object

regr = linear_model.LinearRegression()



# Fit regression model to the training set

regr.fit(X_train, y_train)
# Apply model to the test set

y_pred_test = regr.predict(X_test)
# Comparing true versus predicted values

plt.scatter(y_test, y_pred_test, color='black')

plt.title('Comparing true and predicted values for test set')

plt.xlabel('True values for y')

plt.ylabel('Predicted values for y')



# Model evaluation

print("Root mean squared error = %.4f" % np.sqrt(mean_squared_error(y_test, y_pred_test)))

print('R-squared = %.4f' % r2_score(y_test, y_pred_test))
# Display model parameters

print('Slope = ', regr.coef_[0][0])

print('Intercept = ', regr.intercept_[0])### Step 4: Postprocessing



# Plot outputs

plt.scatter(X_test, y_test,  color='black')

plt.plot(X_test, y_pred_test, color='blue', linewidth=3)

titlestr = 'Predicted Function: y = %.2fX + %.2f' % (regr.coef_[0], regr.intercept_[0])

plt.title(titlestr)

plt.xlabel('X')

plt.ylabel('y')
seed = 1

np.random.seed(seed)

X2 = 0.5*X + np.random.normal(0, 0.04, size=numInstances).reshape(-1,1)

X3 = 0.5*X2 + np.random.normal(0, 0.01, size=numInstances).reshape(-1,1)

X4 = 0.5*X3 + np.random.normal(0, 0.01, size=numInstances).reshape(-1,1)

X5 = 0.5*X4 + np.random.normal(0, 0.01, size=numInstances).reshape(-1,1)



fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(12,9))

ax1.scatter(X, X2, color='black')

ax1.set_xlabel('X')

ax1.set_ylabel('X2')

c = np.corrcoef(np.column_stack((X[:-numTest],X2[:-numTest])).T)

titlestr = 'Correlation between X and X2 = %.4f' % (c[0,1])

ax1.set_title(titlestr)



ax2.scatter(X2, X3, color='black')

ax2.set_xlabel('X2')

ax2.set_ylabel('X3')

c = np.corrcoef(np.column_stack((X2[:-numTest],X3[:-numTest])).T)

titlestr = 'Correlation between X2 and X3 = %.4f' % (c[0,1])

ax2.set_title(titlestr)



ax3.scatter(X3, X4, color='black')

ax3.set_xlabel('X3')

ax3.set_ylabel('X4')

c = np.corrcoef(np.column_stack((X3[:-numTest],X4[:-numTest])).T)

titlestr = 'Correlation between X3 and X4 = %.4f' % (c[0,1])

ax3.set_title(titlestr)



ax4.scatter(X4, X5, color='black')

ax4.set_xlabel('X4')

ax4.set_ylabel('X5')

c = np.corrcoef(np.column_stack((X4[:-numTest],X5[:-numTest])).T)

titlestr = 'Correlation between X4 and X5 = %.4f' % (c[0,1])

ax4.set_title(titlestr)