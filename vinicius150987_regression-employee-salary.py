import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

sns.set(style='white', palette='dark')

warnings.filterwarnings('ignore')

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing Dataset

df = pd.read_csv('/kaggle/input/employee-salary/Employee_Salary.csv')

df
#Analysing dataset with padas profiling

from pandas_profiling import ProfileReport

profile = ProfileReport(df, title='Employee Salary Datasets', html={'style':{'full_width':True}})
profile
#Dataset statistic

df.describe()
#Dataset info

df.info()
#Plotting dataset

fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(1,1,1)

ax.scatter(df['Years of Experience'].values, df['Salary'].values,marker='o', color='r', alpha=1, linewidth=1, 

            edgecolor='k')

ax.set_title('Relationship between Year of Experience x Salary', fontsize=15)

ax.set_xlabel(df.columns[0], fontsize=15)

ax.set_ylabel(df.columns[1], fontsize=15)

ax.grid(b=True, which='major', linestyle='--')

ax.tick_params(axis='both', labelsize=15, labelcolor='k')

plt.tight_layout(0)
#Splitting Data

X=df.drop('Salary', axis=1)

y=df['Salary']
#Splitting the Dataset into the training set and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_train.shape, X_test.shape, y_train.shape,y_test.shape
#Feature scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

X_train = pd.DataFrame(sc_x.fit_transform(X_train), columns=X.columns.values)

X_test = pd.DataFrame(sc_x.transform(X_test), columns=X.columns.values)
## Multiple Linear Regression Regression

from sklearn.linear_model import LinearRegression

lr_regressor = LinearRegression(fit_intercept=True)

lr_regressor.fit(X_train, y_train)



print('Linear Model Coefficient (m): ', lr_regressor.coef_)

print('Linear Model Coefficient (b): ', lr_regressor.intercept_)



# Predicting Test Set

y_pred = lr_regressor.predict(X_test)



plt.scatter(X_train, y_train, color = 'gray')

plt.plot(X_train, lr_regressor.predict(X_train), color = 'red')

plt.ylabel('{} ($)'.format(df.columns[1]))

plt.xlabel('{}'.format(df.columns[0]))

plt.title('Employee Salary (Training dataset)')



plt.scatter(X_test, y_test, color = 'gray')

plt.plot(X_test, lr_regressor.predict(X_test), color = 'red')

plt.ylabel('{} ($)'.format(df.columns[1]))

plt.xlabel('{}'.format(df.columns[0]))

plt.title('Employee Salary (Test dataset)')
plt.scatter(X_test, y_test, color = 'gray')

plt.plot(X_test, lr_regressor.predict(X_test), color = 'red')

plt.ylabel('{} ($)'.format(df.columns[1]))

plt.xlabel('{}'.format(df.columns[0]))

plt.title('Employee Salary (Test dataset)')
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

lr_poly_regressor = LinearRegression()

lr_poly_regressor.fit(X_poly, y_train)



# Predicting Test Set

y_pred = lr_poly_regressor.predict(poly_reg.fit_transform(X_test))



plt.scatter(X_test, y_test, color = 'gray')

plt.scatter(X_test, y_pred, color = 'red')

plt.ylabel('{} ($)'.format(df.columns[1]))

plt.xlabel('{}'.format(df.columns[0]))

plt.title('Employee Salary (Test dataset)')
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

svr_regressor = SVR(kernel = 'rbf')

svr_regressor.fit(X_train, y_train)



# Predicting Test Set

y_pred = svr_regressor.predict(X_test)

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

dt_regressor = DecisionTreeRegressor(random_state=0)

dt_regressor.fit(X_train, y_train)



# Predicting Test Set

y_pred = dt_regressor.predict(X_test)

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

rf_regressor = RandomForestRegressor(n_estimators=300, random_state=0)

rf_regressor.fit(X_train,y_train)



# Predicting Test Set

y_pred = rf_regressor.predict(X_test)

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

ad_regressor = AdaBoostRegressor()

ad_regressor.fit(X_train, y_train)



# Predicting Test Set

y_pred = ad_regressor.predict(X_test)

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

gb_regressor = GradientBoostingRegressor()

gb_regressor.fit(X_train, y_train)



# Predicting Test Set

y_pred = gb_regressor.predict(X_test)

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

xgb_regressor = XGBRegressor()

xgb_regressor.fit(X_train, y_train)



# Predicting Test Set

y_pred = xgb_regressor.predict(X_test)

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, y_pred)

mse = metrics.mean_squared_error(y_test, y_pred)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

r2 = metrics.r2_score(y_test, y_pred)



model_results = pd.DataFrame([['XGB Regressor', mae, mse, rmse, r2]],

               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])



results = results.append(model_results, ignore_index = True)
##Ensemble Voting regressor

from sklearn.ensemble import VotingRegressor

voting_regressor = VotingRegressor(estimators= [('lr', lr_regressor),

                                                  ('lr_poly', lr_poly_regressor),

                                                  ('svr', svr_regressor),

                                                  ('dt', dt_regressor),

                                                  ('rf', rf_regressor),

                                                  ('ad', ad_regressor),

                                                  ('gb', gb_regressor),

                                                  ('xg', xgb_regressor)])



for clf in (lr_regressor,lr_poly_regressor,svr_regressor,dt_regressor,

            rf_regressor, ad_regressor,gb_regressor, xgb_regressor, voting_regressor):

    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    print(clf.__class__.__name__, metrics.r2_score(y_test, y_pred))



# Predicting Test Set

y_pred = voting_regressor.predict(X_test)

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, y_pred)

mse = metrics.mean_squared_error(y_test, y_pred)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

r2 = metrics.r2_score(y_test, y_pred)



model_results = pd.DataFrame([['Ensemble Voting', mae, mse, rmse, r2]],

               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])



results = results.append(model_results, ignore_index = True)  
#The Best Classifier

print('The best regressor is:')

print('{}'.format(results.sort_values(by='R2 Score',ascending=False).head(5)))
#Applying K-fold validation

from sklearn.model_selection import cross_val_score

def display_scores (scores):

    print('Scores:', scores)

    print('Mean:', scores.mean())

    print('Standard:', scores.std())



lin_scores = cross_val_score(estimator=gb_regressor, X=X_train, y=y_train, 

                             scoring= 'neg_mean_squared_error',cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)


