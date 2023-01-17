# Polynomial Regression



# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# Importing the dataset

dataset = pd.read_csv('../input/diabetic-retinopathy/diabetic.csv')

X = dataset.iloc[:, 6:7].values

y = dataset.iloc[:, 7].values



# Splitting the dataset into the Training set and Test set

"""from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""



# Feature Scaling

"""from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)"""



# Fitting Linear Regression to the dataset

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X, y)



# Fitting Polynomial Regression to the dataset

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)

X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)

lin_reg_2 = LinearRegression()

lin_reg_2.fit(X_poly, y)



# Visualising the Linear Regression results

plt.scatter(X, y, color = 'red')

plt.plot(X, lin_reg.predict(X), color = 'blue')

plt.title('Truth or Bluff (Linear Regression)')

plt.show()



# Visualising the Polynomial Regression results

plt.scatter(X, y, color = 'red')

plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')

plt.title('Truth or Bluff (Polynomial Regression)')

plt.show()



# Visualising the Polynomial Regression results (for higher resolution and smoother curve)

X_grid = np.arange(min(X), max(X), 0.1)

X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'red')

plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')

plt.title('Truth or Bluff (Polynomial Regression)')

plt.show()



# Predicting a new result with Linear Regression

lin_reg.predict([[6.5]])



# Predicting a new result with Polynomial Regression

lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
# SVR



# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# Importing the dataset

dataset = pd.read_csv('../input/diabetic-retinopathy/diabetic.csv')

X = dataset.iloc[:, 6:7].values

y = dataset.iloc[:, 6].values



# Splitting the dataset into the Training set and Test set

"""from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""



# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

sc_y = StandardScaler()

X = sc_X.fit_transform(X)

y = sc_y.fit_transform(y.reshape(-1,1))



# Fitting SVR to the dataset

from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')

regressor.fit(X, y)



# Predicting a new result

y_pred = regressor.predict([[6.5]])

y_pred = sc_y.inverse_transform(y_pred)



# Visualising the SVR results

plt.scatter(X, y, color = 'red')

plt.plot(X, regressor.predict(X), color = 'blue')

plt.title('Truth or Bluff (SVR)')

plt.show()



# Visualising the SVR results (for higher resolution and smoother curve)

X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled

X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'red')

plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')

plt.title('Truth or Bluff (SVR)')

plt.show()
# Decision Tree Regression



# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# Importing the dataset

dataset = pd.read_csv('../input/diabetic-retinopathy/diabetic.csv')

X = dataset.iloc[:, 6:7].values

y = dataset.iloc[:, 7].values



# Splitting the dataset into the Training set and Test set

"""from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""



# Feature Scaling

"""from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

sc_y = StandardScaler()

y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""



# Fitting Decision Tree Regression to the dataset

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)

regressor.fit(X, y)



# Predicting a new result

y_pred = regressor.predict([[6.5]])



# Visualising the Decision Tree Regression results (higher resolution)

X_grid = np.arange(min(X), max(X), 0.01)

X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'red')

plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')

plt.title('Truth or Bluff (Decision Tree Regression)')

plt.show()
# Random Forest Regression



# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# Importing the dataset

dataset = pd.read_csv('../input/diabetic-retinopathy/diabetic.csv')

X = dataset.iloc[:, 6:7].values

y = dataset.iloc[:, 7].values



# Splitting the dataset into the Training set and Test set

"""from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""



# Feature Scaling

"""from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

sc_y = StandardScaler()

y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""



# Fitting Random Forest Regression to the dataset

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

regressor.fit(X, y)



# Predicting a new result

y_pred = regressor.predict([[6.5]])



# Visualising the Random Forest Regression results (higher resolution)

X_grid = np.arange(min(X), max(X), 0.01)

X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'red')

plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')

plt.title('Truth or Bluff (Random Forest Regression)')

plt.show()