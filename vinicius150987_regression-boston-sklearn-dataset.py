import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

sns.set(style='white', palette='deep')

width=0.35

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing dataset

from sklearn.datasets import load_boston

boston = load_boston()



for i in np.arange(len(boston)):

    print(list(boston.keys())[i])



data = np.c_[boston['data'], boston['target']]

columns = np.append(boston['feature_names'], 'MEDV')



df= pd.DataFrame(data, columns= columns)

df.head()
#Finding the meaning of the columns

df.columns

boston['DESCR'][

        boston['DESCR'].find(df.columns[0]):boston['DESCR'].find(df.columns[0])+

        len(df.columns[0])]



text_list = boston['DESCR'].split('\n')



for i in np.arange(len(text_list)):

    print(i, text_list[i])

    

text_list = text_list[12:25]

text_list = [text_list[i].strip('- ') for i in np.arange(len(text_list))]

text_list = [text_list[i].strip(df.columns[i]).strip() for i in np.arange(len(text_list))]



columns_meaning = pd.DataFrame(text_list, index=boston['feature_names'], columns=['Description'])

target_meaning = pd.DataFrame(["Median value of owner-occupied homes in $1000's"], index=['MEDV'],columns=['Description'])

columns_meaning = pd.concat([columns_meaning,target_meaning])

columns_meaning
#Looking for null values

null_values = (df.isnull().sum()/len(df))*100

null_values = pd.DataFrame(null_values, columns=['% of Null Values'])

null_values
#The maximum and minimum values by columns

describe = df.describe().loc[['min','max']]

describe
## Histograms

df2 = df.loc[:,boston['feature_names']]



fig = plt.figure(figsize=(10, 10))

plt.suptitle('Histograms of Numerical Columns', fontsize=20)

for i in range(df2.shape[1]):

    plt.subplot(6, 3, i + 1)

    f = plt.gca()

    f.set_title(df2.columns.values[i])



    vals = np.size(df2.iloc[:, i].unique())

    if vals >= 100:

        vals = 100

    

    plt.hist(df2.iloc[:, i], bins=vals, color='#3F5D7D')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
## Correlation with independent Variable (Note: Models like RF are not linear like these)

df.columns

df2.corrwith(df['MEDV']).plot.bar(

        figsize = (10, 10), title = "Correlation with MEDV", fontsize = 15,

        rot = 45, grid = True)
#Define X and y

X = df.loc[:,boston['feature_names']]

y = df.loc[:,'MEDV']
#Splitting the Dataset into the training set and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
#Feature scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

X_train = pd.DataFrame(sc_x.fit_transform(X_train), columns=X.columns.values)

X_test = pd.DataFrame(sc_x.transform(X_test), columns=X.columns.values)
#### Model Building ####

### Comparing Models



## Multiple Linear Regression Regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)



# Predicting Test Set

y_pred = regressor.predict(X_test)

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, y_pred)

mse = metrics.mean_squared_error(y_test, y_pred)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

r2 = metrics.r2_score(y_test, y_pred)



results = pd.DataFrame([['Multiple Linear Regression', mae, mse, rmse, r2]],

               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])
## Polynomial Regressor

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 2)

X_poly = poly_reg.fit_transform(X_train)

regressor = LinearRegression()

regressor.fit(X_poly, y_train)



# Predicting Test Set

y_pred = regressor.predict(poly_reg.fit_transform(X_test))

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, y_pred)

mse = metrics.mean_squared_error(y_test, y_pred)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

r2 = metrics.r2_score(y_test, y_pred)



model_results = pd.DataFrame([['Polynomial Regression', mae, mse, rmse, r2]],

               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])



results = results.append(model_results, ignore_index = True)
## Suport Vector Regression 

'Necessary Standard Scaler '

from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')

regressor.fit(X_train, y_train)



# Predicting Test Set

y_pred = regressor.predict(X_test)

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, y_pred)

mse = metrics.mean_squared_error(y_test, y_pred)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

r2 = metrics.r2_score(y_test, y_pred)



model_results = pd.DataFrame([['Support Vector RBF', mae, mse, rmse, r2]],

               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])



results = results.append(model_results, ignore_index = True)
## Decision Tree Regression

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)

regressor.fit(X_train, y_train)



# Predicting Test Set

y_pred = regressor.predict(X_test)

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, y_pred)

mse = metrics.mean_squared_error(y_test, y_pred)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

r2 = metrics.r2_score(y_test, y_pred)



model_results = pd.DataFrame([['Decision Tree Regression', mae, mse, rmse, r2]],

               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])



results = results.append(model_results, ignore_index = True)
## Random Forest Regression

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=300, random_state=0)

regressor.fit(X_train,y_train)



# Predicting Test Set

y_pred = regressor.predict(X_test)

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, y_pred)

mse = metrics.mean_squared_error(y_test, y_pred)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

r2 = metrics.r2_score(y_test, y_pred)



model_results = pd.DataFrame([['Random Forest Regression', mae, mse, rmse, r2]],

               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])



results = results.append(model_results, ignore_index = True)
## Ada Boosting

from sklearn.ensemble import AdaBoostRegressor

regressor = AdaBoostRegressor()

regressor.fit(X_train, y_train)



# Predicting Test Set

y_pred = regressor.predict(X_test)

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, y_pred)

mse = metrics.mean_squared_error(y_test, y_pred)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

r2 = metrics.r2_score(y_test, y_pred)



model_results = pd.DataFrame([['AdaBoost Regressor', mae, mse, rmse, r2]],

               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])



results = results.append(model_results, ignore_index = True)
##Gradient Boosting

from sklearn.ensemble import GradientBoostingRegressor

regressor = GradientBoostingRegressor()

regressor.fit(X_train, y_train)



# Predicting Test Set

y_pred = regressor.predict(X_test)

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, y_pred)

mse = metrics.mean_squared_error(y_test, y_pred)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

r2 = metrics.r2_score(y_test, y_pred)



model_results = pd.DataFrame([['GradientBoosting Regressor', mae, mse, rmse, r2]],

               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])



results = results.append(model_results, ignore_index = True)
##Xg Boosting

from xgboost import XGBRegressor

regressor = XGBRegressor()

regressor.fit(X_train, y_train)



# Predicting Test Set

y_pred = regressor.predict(X_test)

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, y_pred)

mse = metrics.mean_squared_error(y_test, y_pred)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

r2 = metrics.r2_score(y_test, y_pred)



model_results = pd.DataFrame([['XGB Regressor', mae, mse, rmse, r2]],

               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])



results = results.append(model_results, ignore_index = True)
#The best model

results.sort_values(by='RMSE', ascending=True)