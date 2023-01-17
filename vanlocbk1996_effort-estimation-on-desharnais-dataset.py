import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR
df = pd.read_csv('../input/desharnais-dataset/02.desharnais.csv')

df.head(5)
df.info()
df.describe()
colormap = plt.cm.viridis

plt.figure(figsize=(10,10))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.set(font_scale=1.05)

sns.heatmap(df.drop(['id'], axis=1).astype(float).corr(),linewidths=0.1,vmax=1.0, square=True,cmap=colormap, linecolor='white', annot=True)
features = [ 'TeamExp', 'ManagerExp', 'YearEnd', 'Length', 'Transactions', 'Entities',

        'PointsNonAdjust', 'Adjustment', 'PointsAjust']



max_corr_features = ['Length', 'Transactions', 'Entities','PointsNonAdjust','PointsAjust']



X = df[max_corr_features]

y = df['Effort']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=30)

neigh = KNeighborsRegressor(n_neighbors=3, weights='uniform')

neigh.fit(X_train, y_train) 

predict = neigh.predict(X_test)

print("Root mean square: ", np.sqrt(mean_squared_error(y_test, predict)))

print("Mean Absolute error", mean_absolute_error(y_test, predict))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=22)

regressor = LinearRegression()

regressor.fit(X_train, y_train)

predict = regressor.predict(X_test)

print("Root mean square: ", np.sqrt(mean_squared_error(y_test, predict)))

print("Mean Absolute error", mean_absolute_error(y_test, predict))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

for i in np.arange(100,1100,100):

    regressor = RandomForestRegressor(n_estimators=i,max_features=5)

    regressor.fit(X_train, y_train)

    predict = regressor.predict(X_test)

    print("Root mean square: ", np.sqrt(mean_squared_error(y_test, predict)))

    print("Mean Absolute error", mean_absolute_error(y_test, predict))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

parameters = {'kernel':('linear', 'rbf'), 'C':[1,2,3,4,5,6,7,8,9,10], 'gamma':('auto', 'scale')}



svr = SVR()

LinearSVC = GridSearchCV(svr, parameters, cv=3)

LinearSVC.fit(X_train, y_train)

predict = LinearSVC.predict(X_test)

print("Root mean square: ", np.sqrt(mean_squared_error(y_test, predict)))

print("Mean Absolute error", mean_absolute_error(y_test, predict))