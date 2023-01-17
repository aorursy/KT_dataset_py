# data visualization

import matplotlib.pyplot as plt

plt.rc('figure', figsize=(14, 9))



# data representation and manipulation

import pandas as pd



# we'll import several things from various parts of sklearn.

# sklearn is the library that provides us machine learning models

# and various utility functions for machine learning

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor



from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split



# load the dataset

df = pd.read_csv('/kaggle/input/lemonadesales/lemonade_clean.csv')

df
X = df[['temperature']]

y = df[['sales']]
X_train, X_test, y_train, y_test = train_test_split(X, y)
# create the model

lr = LinearRegression()

# fit the model on the training data

lr.fit(X_train, y_train)



m = lr.coef_

b = lr.intercept_



print('m =', m, 'b =', b)
print('y = {:.2f}x + {:.2f}'.format(m[0, 0], b[0]))
temperature = X_train

actual_sales = y_train

lr_predicted_sales = lr.predict(X_train)



plt.scatter(temperature, actual_sales, label='Actual')

plt.scatter(temperature, lr_predicted_sales, label='Predicted')

plt.xlabel('Temperature')

plt.ylabel('Sales')

plt.legend()

plt.title('Sales: Actual vs Linear Regression Predictions for Training Data')
mean_squared_error(actual_sales, lr_predicted_sales)
knr = KNeighborsRegressor(n_neighbors=2)

knr.fit(X_train, y_train)



knr_predicted_sales = knr.predict(X_train)
plt.scatter(temperature, actual_sales, label='Actual')

plt.scatter(temperature, knr_predicted_sales, label='Predictions')

plt.legend()

plt.xlabel('Temperature')

plt.ylabel('Sales')

plt.title('Sales: Actual vs K-Neighbors Predictions for Training Data')
mean_squared_error(actual_sales, knr_predicted_sales)
knr_test_predictions = knr.predict(X_test)

lr_test_predictions = lr.predict(X_test)
plt.scatter(X_test, y_test, label='Actual')

plt.scatter(X_test, lr_test_predictions, label='Predictions')

plt.legend()

plt.xlabel('Temperature')

plt.ylabel('Sales')

plt.title('Sales: Actual vs Linear Regression Predictions for Test Data')
plt.scatter(X_test, y_test, label='Actual')

plt.scatter(X_test, knr_test_predictions, label='Predictions')

plt.legend()

plt.xlabel('Temperature')

plt.ylabel('Sales')

plt.title('Sales: Actual vs K-Neighbors Predictions for Test Data')
knr_mse = mean_squared_error(y_test, knr_test_predictions)

lr_mse = mean_squared_error(y_test, lr_test_predictions)



print('k-neighbors test mean squared error: ', knr_mse)

print('linear regression test mean squared error: ', lr_mse)
lr_test_residuals = y_test - lr_test_predictions

knr_test_residuals = y_test - knr_test_predictions
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)



ax1.scatter(y_test, lr_test_residuals)

ax1.set(ylabel='Residual ($y - \hat{y}$)', xlabel='Sales', title='Linear Regression')



ax2.scatter(y_test, knr_test_residuals)

ax2.set(title='K-Neighbors', xlabel='Sales')



fig.suptitle('Test Data Set Residuals')