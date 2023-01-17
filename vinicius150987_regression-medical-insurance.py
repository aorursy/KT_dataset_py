import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

sns.set(style='white', palette='deep')

warnings.filterwarnings('ignore')

%matplotlib inline





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing dataset

df = pd.read_csv('/kaggle/input/insurance/insurance.csv')

df.head()
#Analysing dataset with traditional way

statistical = df.describe()

statistical
#Null values

null_values = (df.isnull().sum()/len(df))*100

null_values = pd.DataFrame(null_values, columns=['% of Nulll Values'])

null_values
#Analysing dataset with padas profiling

from pandas_profiling import ProfileReport

profile = ProfileReport(df, title='Medical Cost Personal Datasets', html={'style':{'full_width':True}})

profile
#Analysing insurance by age - Pivot Table

df.columns

bins = np.arange(15,71,5)

bins2 = bins.astype(str)

age = pd.cut(df['age'],bins)



count1 = 0

count2 = 1

labels = np.array([])



while count2 < len(np.unique(age)) + 1:

    labels = np.append(labels,'['+bins2[count1] + '-' + bins2[count2]+']')

    count1+=1

    count2+=1

pivot_age_mean = df.pivot_table('charges', index=age,columns='sex')

pivot_age_mean
#Plotting pivot_age_mean

ind = np.arange(len(labels))

fig=plt.figure(figsize=(10,10))

fig.suptitle('Insurance by Genre', fontsize=15, y=1.05)

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)

pivot_age_mean['male'].plot(ax=ax1, marker='o')

pivot_age_mean['female'].plot(ax=ax2, color='orange', sharey=ax1, marker='o')

ax1.set_xlabel('Age', fontsize=15)

ax1.set_ylabel('Insurance', fontsize=15)

ax1.grid(b=True, which='major', linestyle='--')

ax1.set_title('Insurance for Male by Age', fontsize=15)

ax1.tick_params(axis='both', labelsize=15, labelcolor='k')

ax1.set_xticks(ind)

ax1.tick_params(axis='x', labelsize=15, labelcolor='k', rotation=90)

ax1.set_xticklabels(labels, fontsize=15)



ax2.set_xlabel('Age', fontsize=15)

ax2.set_ylabel('Insurance', fontsize=15)

ax2.grid(b=True, which='major', linestyle='--')

ax2.set_title('Insurance for Female by Age', fontsize=15)

ax2.tick_params(axis='both', labelsize=15, labelcolor='k')

ax2.tick_params(axis='both', labelsize=15, labelcolor='k')

ax2.set_xticks(ind)

ax2.tick_params(axis='x', labelsize=15, labelcolor='k', rotation=90)

ax2.set_xticklabels(labels, fontsize=15)

plt.tight_layout()
#Analysing insurance by age and smoke - Pivot Table

df.columns 

pivot_smoke_mean = df.pivot_table('charges', index=age, columns=['sex','smoker'])



fig = plt.figure(figsize=(10,10))

fig.suptitle('Insurance by Smokers', y=1.05)

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)

pivot_smoke_mean['male'].plot(ax=ax1, marker='o')

pivot_smoke_mean['female'].plot(ax=ax2, sharey=ax1, marker='o')

ax1.set_xlabel('Age', fontsize=15)

ax1.set_xticks(ind)

ax1.set_xticklabels(labels, fontsize=15)

ax1.grid(b=True, which='major', linestyle='--')

ax1.legend(['No', 'Yes'], title='Smoker')

ax1.set_title('Insurance for Male by Smokers', fontsize=15)

ax1.tick_params(axis='x', labelsize=15, rotation=90)

ax1.tick_params(axis='y', labelsize=15 )

ax1.set_ylabel('Insurance', fontsize=15)



ax2.set_xlabel('Age', fontsize=15)

ax2.set_xticks(ind)

ax2.set_xticklabels(labels, fontsize=15)

ax2.grid(b=True, which='major', linestyle='--')

ax2.legend(['No', 'Yes'], title='Smoker')

ax2.set_title('Insurance for Female by Smokers', fontsize=15)

ax2.tick_params(axis='x', labelsize=15, rotation=90)

ax2.tick_params(axis='y', labelsize=15 )

ax2.set_ylabel('Insurance', fontsize=15)

plt.tight_layout()
#Analysing insurance by regions - Pivot Table

pivot_region_mean = df.pivot_table('charges', index=['sex', age], columns=['region', 'smoker'])

fig = plt.figure(figsize=(10,10))

fig.suptitle('Insurance by Regions', y=1.05)

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)



pivot_region_mean.loc['male', pivot_region_mean.columns.levels[0]].plot(ax=ax1, marker='o')

pivot_region_mean.loc['female', pivot_region_mean.columns.levels[0]].plot(ax=ax2, sharey=ax1, marker='o')

ax1.set(title='Insurance for Male by Regions', xlabel='Age', xticklabels=labels, xticks=ind,

        ylabel='Insurance')

ax1.grid(b=True, which='major', linestyle='--')

ax1.legend(loc='best')

ax1.tick_params(axis='x', rotation=90)



ax2.set(title='Insurance for Female by Regions', xlabel='Age', xticklabels=labels, xticks=ind,

        ylabel='Insurance')

ax2.grid(b=True, which='major', linestyle='--')

ax2.legend(loc='best')

ax2.tick_params(axis='x', rotation=90)

plt.tight_layout()
#Analysing insurance by BMI - Pivot Table

df['bmi'].min()

df['bmi'].max()

bins_bmi = np.arange(10,70,10)

bins_bmi2 = pd.cut(df['bmi'], bins_bmi)

bins_bmi_string = np.sort(bins_bmi2.unique().astype(str))

ind = np.arange(len(bins_bmi_string))

pivot_bmi_mean = df.pivot_table('charges', index=bins_bmi2, columns='sex')

pivot_bmi_mean



pivot_bmi_mean.plot(marker='o')

plt.xlabel('Body Mass Index')

plt.xticks(ind, bins_bmi_string)

plt.title('Insurance by Body Mass Index')

plt.grid(b=True, which='major', linestyle='--')

plt.ylabel('Insurance')
#Analysing insurance by Children- Pivot Table

pivot_children_mean = df.pivot_table('charges', index='children', columns='sex')

pivot_children_mean

ind = np.arange(len(pivot_children_mean))



pivot_children_mean.plot(marker='o')

plt.xlabel('Number of Children')

plt.xticks(ind, pivot_children_mean.index)

plt.title('Insurance by Children')

plt.grid(b=True, which='major', linestyle='--')

plt.ylabel('Insurance')
## Correlation with independent Variable (Note: Models like RF are not linear like these)

df2 = df.drop(['charges'], axis=1)

df2.corrwith(df.charges).plot.bar(

        figsize = (10, 10), title = "Correlation with Charges", fontsize = 10,

        rot = 45, grid = True)
#Splitting dataset into X and y

X = df.drop('charges', axis=1)

y = df['charges']
#Getting dummies variables

X = pd.get_dummies(X) 
#Avoiding dummies trap

X.columns

X = X.drop(['sex_male','smoker_yes','region_northeast' ], axis=1)
#Splitting the Dataset into the training set and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=0)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
#Feature scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

X_train = pd.DataFrame(sc_x.fit_transform(X_train), columns=X.columns.values)

X_test = pd.DataFrame(sc_x.transform(X_test), columns=X.columns.values)
#### Model Building ####

### Comparing Models



## Multiple Linear Regression Regression

from sklearn.linear_model import LinearRegression

lr_regressor = LinearRegression()

lr_regressor.fit(X_train, y_train)



# Predicting Test Set

y_pred = lr_regressor.predict(X_test)

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
#Analysing results

results
#The Best Regressor

print('The best regressor is:')

print('{}'.format(results.sort_values(by='R2 Score',ascending=False).head(5)))
#Applying K-fold validation

from sklearn.model_selection import cross_val_score

def display_scores (scores):

    print('Scores:', scores)

    print('Mean:', scores.mean())

    print('Standard:', scores.std())



lin_scores = cross_val_score(estimator=xgb_regressor, X=X_train, y=y_train, 

                             scoring= 'neg_mean_squared_error',cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)