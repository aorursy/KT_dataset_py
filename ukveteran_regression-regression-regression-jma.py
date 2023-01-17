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
X_train2 = np.column_stack((X[:-numTest],X2[:-numTest]))

X_test2 = np.column_stack((X[-numTest:],X2[-numTest:]))

X_train3 = np.column_stack((X[:-numTest],X2[:-numTest],X3[:-numTest]))

X_test3 = np.column_stack((X[-numTest:],X2[-numTest:],X3[-numTest:]))

X_train4 = np.column_stack((X[:-numTest],X2[:-numTest],X3[:-numTest],X4[:-numTest]))

X_test4 = np.column_stack((X[-numTest:],X2[-numTest:],X3[-numTest:],X4[-numTest:]))

X_train5 = np.column_stack((X[:-numTest],X2[:-numTest],X3[:-numTest],X4[:-numTest],X5[:-numTest]))

X_test5 = np.column_stack((X[-numTest:],X2[-numTest:],X3[-numTest:],X4[-numTest:],X5[-numTest:]))
regr2 = linear_model.LinearRegression()

regr2.fit(X_train2, y_train)



regr3 = linear_model.LinearRegression()

regr3.fit(X_train3, y_train)



regr4 = linear_model.LinearRegression()

regr4.fit(X_train4, y_train)



regr5 = linear_model.LinearRegression()

regr5.fit(X_train5, y_train)
y_pred_train = regr.predict(X_train)

y_pred_test = regr.predict(X_test)

y_pred_train2 = regr2.predict(X_train2)

y_pred_test2 = regr2.predict(X_test2)

y_pred_train3 = regr3.predict(X_train3)

y_pred_test3 = regr3.predict(X_test3)

y_pred_train4 = regr4.predict(X_train4)

y_pred_test4 = regr4.predict(X_test4)

y_pred_train5 = regr5.predict(X_train5)

y_pred_test5 = regr5.predict(X_test5)
import pandas as pd

import matplotlib.pyplot as plt



columns = ['Model', 'Train error', 'Test error', 'Sum of Absolute Weights']

model1 = "%.2f X + %.2f" % (regr.coef_[0][0], regr.intercept_[0])

values1 = [ model1, np.sqrt(mean_squared_error(y_train, y_pred_train)), 

           np.sqrt(mean_squared_error(y_test, y_pred_test)),

           np.absolute(regr.coef_[0]).sum() + np.absolute(regr.intercept_[0])]



model2 = "%.2f X + %.2f X2 + %.2f" % (regr2.coef_[0][0], regr2.coef_[0][1], regr2.intercept_[0])

values2 = [ model2, np.sqrt(mean_squared_error(y_train, y_pred_train2)), 

           np.sqrt(mean_squared_error(y_test, y_pred_test2)),

           np.absolute(regr2.coef_[0]).sum() + np.absolute(regr2.intercept_[0])]



model3 = "%.2f X + %.2f X2 + %.2f X3 + %.2f" % (regr3.coef_[0][0], regr3.coef_[0][1], 

                                                regr3.coef_[0][2], regr3.intercept_[0])

values3 = [ model3, np.sqrt(mean_squared_error(y_train, y_pred_train3)), 

           np.sqrt(mean_squared_error(y_test, y_pred_test3)),

           np.absolute(regr3.coef_[0]).sum() + np.absolute(regr3.intercept_[0])]



model4 = "%.2f X + %.2f X2 + %.2f X3 + %.2f X4 + %.2f" % (regr4.coef_[0][0], regr4.coef_[0][1], 

                                        regr4.coef_[0][2], regr4.coef_[0][3], regr4.intercept_[0])

values4 = [ model4, np.sqrt(mean_squared_error(y_train, y_pred_train4)), 

           np.sqrt(mean_squared_error(y_test, y_pred_test4)),

           np.absolute(regr4.coef_[0]).sum() + np.absolute(regr4.intercept_[0])]



model5 = "%.2f X + %.2f X2 + %.2f X3 + %.2f X4 + %.2f X5 + %.2f" % (regr5.coef_[0][0], 

                                        regr5.coef_[0][1], regr5.coef_[0][2], 

                                        regr5.coef_[0][3], regr5.coef_[0][4], regr5.intercept_[0])

values5 = [ model5, np.sqrt(mean_squared_error(y_train, y_pred_train5)), 

           np.sqrt(mean_squared_error(y_test, y_pred_test5)),

           np.absolute(regr5.coef_[0]).sum() + np.absolute(regr5.intercept_[0])]



results = pd.DataFrame([values1, values2, values3, values4, values5], columns=columns)



plt.plot(results['Sum of Absolute Weights'], results['Train error'], 'ro-')

plt.plot(results['Sum of Absolute Weights'], results['Test error'], 'k*--')

plt.legend(['Train error', 'Test error'])

plt.xlabel('Sum of Absolute Weights')

plt.ylabel('Error rate')



results
from sklearn import linear_model



ridge = linear_model.Ridge(alpha=0.4)

ridge.fit(X_train5, y_train)

y_pred_train_ridge = ridge.predict(X_train5)

y_pred_test_ridge = ridge.predict(X_test5)



model6 = "%.2f X + %.2f X2 + %.2f X3 + %.2f X4 + %.2f X5 + %.2f" % (ridge.coef_[0][0], 

                                        ridge.coef_[0][1], ridge.coef_[0][2], 

                                        ridge.coef_[0][3], ridge.coef_[0][4], ridge.intercept_[0])

values6 = [ model6, np.sqrt(mean_squared_error(y_train, y_pred_train_ridge)), 

           np.sqrt(mean_squared_error(y_test, y_pred_test_ridge)),

           np.absolute(ridge.coef_[0]).sum() + np.absolute(ridge.intercept_[0])]



ridge_results = pd.DataFrame([values6], columns=columns, index=['Ridge'])

pd.concat([results, ridge_results])
from sklearn import linear_model



lasso = linear_model.Lasso(alpha=0.02)

lasso.fit(X_train5, y_train)

y_pred_train_lasso = lasso.predict(X_train5)

y_pred_test_lasso = lasso.predict(X_test5)



model7 = "%.2f X + %.2f X2 + %.2f X3 + %.2f X4 + %.2f X5 + %.2f" % (lasso.coef_[0], 

                                        lasso.coef_[1], lasso.coef_[2], 

                                        lasso.coef_[3], lasso.coef_[4], lasso.intercept_[0])

values7 = [ model7, np.sqrt(mean_squared_error(y_train, y_pred_train_lasso)), 

           np.sqrt(mean_squared_error(y_test, y_pred_test_lasso)),

           np.absolute(lasso.coef_[0]).sum() + np.absolute(lasso.intercept_[0])]



lasso_results = pd.DataFrame([values7], columns=columns, index=['Lasso'])

pd.concat([results, ridge_results, lasso_results])
from sklearn import linear_model



ridge = linear_model.RidgeCV(cv=5,alphas=[0.2, 0.4, 0.6, 0.8, 1.0])

ridge.fit(X_train5, y_train)

y_pred_train_ridge = ridge.predict(X_train5)

y_pred_test_ridge = ridge.predict(X_test5)



model6 = "%.2f X + %.2f X2 + %.2f X3 + %.2f X4 + %.2f X5 + %.2f" % (ridge.coef_[0][0], 

                                        ridge.coef_[0][1], ridge.coef_[0][2], 

                                        ridge.coef_[0][3], ridge.coef_[0][4], ridge.intercept_[0])

values6 = [ model6, np.sqrt(mean_squared_error(y_train, y_pred_train_ridge)), 

           np.sqrt(mean_squared_error(y_test, y_pred_test_ridge)),

           np.absolute(ridge.coef_[0]).sum() + np.absolute(ridge.intercept_[0])]

print("Selected alpha = %.2f" % ridge.alpha_)



ridge_results = pd.DataFrame([values6], columns=columns, index=['RidgeCV'])

pd.concat([results, ridge_results])
from sklearn import linear_model



lasso = linear_model.LassoCV(cv=5, alphas=[0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 1.0])

lasso.fit(X_train5, y_train.reshape(y_train.shape[0]))

y_pred_train_lasso = lasso.predict(X_train5)

y_pred_test_lasso = lasso.predict(X_test5)



model7 = "%.2f X + %.2f X2 + %.2f X3 + %.2f X4 + %.2f X5 + %.2f" % (lasso.coef_[0], 

                                        lasso.coef_[1], lasso.coef_[2], 

                                        lasso.coef_[3], lasso.coef_[4], lasso.intercept_)

values7 = [ model7, np.sqrt(mean_squared_error(y_train, y_pred_train_lasso)), 

           np.sqrt(mean_squared_error(y_test, y_pred_test_lasso)),

           np.absolute(lasso.coef_[0]).sum() + np.absolute(lasso.intercept_)]

print("Selected alpha = %.2f" % lasso.alpha_)



lasso_results = pd.DataFrame([values7], columns=columns, index=['LassoCV'])

pd.concat([results, ridge_results, lasso_results])