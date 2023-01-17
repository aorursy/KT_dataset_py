# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Loading our Data

data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
#Checking the Features of our Data

data.columns
#Lets get that "Id" column separate

test_Id = test['Id']

data_Id = data['Id']
#Check if we have any Null Values in our Train Data

data.isnull().sum().sort_values(ascending=False).head(20)
data.drop(['Id','PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage', 'GarageCond','GarageType',

       'GarageYrBlt', 'GarageFinish', 'GarageQual', 'BsmtExposure', 'BsmtFinType2', 'BsmtCond', 'BsmtQual', 'MasVnrType'],

      axis = 1, inplace = True)
data.isnull().sum().sort_values(ascending=False).head(5)
data.describe()
#Check if we have any Null Values in our Test Data

test.isnull().sum().sort_values(ascending=False).head(30)
#Same method we apply here too!

test.drop(['Id','PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage', 'GarageCond','GarageType',

       'GarageYrBlt', 'GarageFinish', 'GarageQual', 'BsmtExposure', 'BsmtFinType2', 'BsmtCond', 'BsmtQual', 'MasVnrType'],

      axis = 1, inplace = True)
test.isnull().sum().sort_values(ascending=False).head(20)
num = data.select_dtypes(exclude=['object'])

obj = data.select_dtypes(exclude=['int64', 'float64'])

num_test = test.select_dtypes(exclude=['object'])

obj_test = test.select_dtypes(exclude=['int64', 'float64'])
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

imputed_num = pd.DataFrame(my_imputer.fit_transform(num))

imputed_num.columns = num.columns

imputed_num.isnull().sum().sort_values(ascending=False).head(5)
my_imputer_test = SimpleImputer()

imputed_num_test = pd.DataFrame(my_imputer_test.fit_transform(num_test))

imputed_num_test.columns = num_test.columns

imputed_num_test.isnull().sum().sort_values(ascending=False).head(5)
#Encoding the train cat variables

s = (obj.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
from sklearn.preprocessing import LabelEncoder

label_obj = obj.copy()

label_encoder = LabelEncoder()

for col in object_cols:

    label_obj[col] = label_encoder.fit_transform(obj[col].astype(str))
label_obj
#Encoding the test cat variables

s_test = (obj_test.dtypes == 'object')

object_cols_test = list(s_test[s_test].index)



print("Test Categorical variables:")

print(object_cols_test)
label_obj_test = obj_test.copy()

label_encoder_test = LabelEncoder()

for col in object_cols_test:

    label_obj_test[col] = label_encoder_test.fit_transform(obj_test[col].astype(str))
label_obj_test
label_obj.isnull().sum().sort_values(ascending=False).head()
label_obj_test.isnull().sum().sort_values(ascending=False).head()
cleaned_train = pd.concat([imputed_num, label_obj], axis =1)

cleaned_test = pd.concat([imputed_num_test, label_obj_test], axis =1)
y = cleaned_train['SalePrice']

X = cleaned_train.drop(['SalePrice'], axis = 1)
print(y.shape)

print(X.shape)

print(cleaned_test.shape)
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="white")

corr = cleaned_train.corr()

f, ax = plt.subplots(figsize=(20, 20))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.set(style="white")

corrmat = cleaned_train.corr()

top_corr_features = corrmat.index[(corrmat["SalePrice"])>0.5]

plt.figure(figsize=(10,10))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

g = sns.heatmap(cleaned_train[top_corr_features].corr(), annot=True, cmap='copper', 

                vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
corr['SalePrice'].sort_values(ascending=False).head(15)
from scipy import stats

z1 = np.abs(stats.zscore(cleaned_train))

print(z1)



threshold = 3

print(np.where(z1 > 3))

cleaned_train_2 = cleaned_train.copy()

cleaned_train_OL = cleaned_train_2[(z1 < 3).all(axis=1)].reset_index(drop=True)
cleaned_train_OL.shape
sns.set(style="white")

corr_OL = cleaned_train_OL.corr()

f, ax = plt.subplots(figsize=(20, 20))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.set(style="white")

corrmat_OL = cleaned_train_OL.corr()

top_corr_features = corrmat_OL.index[(corrmat_OL["SalePrice"])>0.5]

plt.figure(figsize=(10,10))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

g = sns.heatmap(cleaned_train_OL[top_corr_features].corr(), annot=True, cmap='copper', 

                vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
print("Top correlated features with our target variable Without Removing Outliers")

print(corr['SalePrice'].sort_values(ascending=False).head(15)) 

print("_______________________________")

print("Top correlated features with our target variable After Removing Outliers")

print(corr_OL['SalePrice'].sort_values(ascending=False).head(15))
sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=False)

sns.despine(left=True)

sns.distplot(cleaned_train['SalePrice'], kde=True, color="m", ax=axes[0])

sns.distplot(cleaned_train_OL['SalePrice'], kde=True, color="green", ax=axes[1])

axes[0].set_title("Without Removing Outliers", size= 20)

axes[1].set_title("After Removing Outliers", size= 20)

plt.setp(axes, yticks=[])

plt.tight_layout()
cleaned_train['OverallQual']
fig, (ax1, ax2) = plt.subplots(figsize = (20,20), nrows=2, sharey=False)



sns.boxplot(cleaned_train['OverallQual'], cleaned_train['SalePrice'], palette=["m", "g"],  ax=ax1)

ax1.set_title("Without Removing Outliers", size = 20)



sns.boxplot(cleaned_train_OL['OverallQual'],cleaned_train_OL['SalePrice'], palette=["m", "g"],ax=ax2)

ax2.set_title("After Removing Outliers", size = 20)
#Analysis after removing outliers - GrLivArea

cleaned_train['GrLivArea'].head()
sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True)

sns.despine(left=True)

sns.distplot(cleaned_train['GrLivArea'], kde=True, color="m", ax=axes[0])

sns.distplot(cleaned_train_OL['GrLivArea'], kde=True, color="green", ax=axes[1])

axes[0].set_title("Without Removing Outliers", size= 20)

axes[1].set_title("After Removing Outliers", size= 20)

plt.setp(axes, yticks=[])

plt.tight_layout()
fig, (ax1, ax2) = plt.subplots(figsize = (20,10), ncols=2, sharey=False)



sns.scatterplot(cleaned_train['GrLivArea'], cleaned_train['SalePrice'],  ax=ax1)

sns.regplot(cleaned_train['GrLivArea'], cleaned_train['SalePrice'], color ='m', ax=ax1)

ax1.set_title("Without Removing Outliers", size = 20)



sns.scatterplot(cleaned_train_OL['GrLivArea'],cleaned_train_OL['SalePrice'], ax=ax2)

sns.regplot(cleaned_train_OL['GrLivArea'],cleaned_train_OL['SalePrice'],color ='g', ax=ax2)

ax2.set_title("After Removing Outliers", size = 20)
cleaned_train['GarageCars'].unique()
fig, (ax1, ax2) = plt.subplots(figsize = (20,20), nrows=2, sharey=False)



sns.swarmplot(cleaned_train['GarageCars'], cleaned_train['SalePrice'],  palette="Set2", ax=ax1)

ax1.set_title("Without Removing Outliers", size = 20)



sns.swarmplot(cleaned_train_OL['GarageCars'],cleaned_train_OL['SalePrice'], palette="Set2", ax=ax2)

ax2.set_title("After Removing Outliers", size = 20)
cleaned_train['GarageArea'].head()
sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True)

sns.despine(left=True)

sns.distplot(cleaned_train['GarageArea'], kde=True, color="m", ax=axes[0])

sns.distplot(cleaned_train_OL['GarageArea'], kde=True, color="green", ax=axes[1])

axes[0].set_title("Without Removing Outliers", size= 20)

axes[1].set_title("After Removing Outliers", size= 20)

plt.setp(axes, yticks=[])

plt.tight_layout()
fig, (ax1, ax2) = plt.subplots(figsize = (20,10), ncols=2, sharey=False)



sns.scatterplot(cleaned_train['GarageArea'], cleaned_train['SalePrice'],  ax=ax1)

sns.regplot(cleaned_train['GarageArea'], cleaned_train['SalePrice'], color ='m', ax=ax1)

ax1.set_title("Without Removing Outliers", size = 20)



sns.scatterplot(cleaned_train_OL['GarageArea'],cleaned_train_OL['SalePrice'], ax=ax2)

sns.regplot(cleaned_train_OL['GarageArea'],cleaned_train_OL['SalePrice'],color ='g', ax=ax2)

ax2.set_title("After Removing Outliers", size = 20)

cleaned_train['TotalBsmtSF'].head()
sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True)

sns.despine(left=True)

sns.distplot(cleaned_train['TotalBsmtSF'], kde=True, color="m", ax=axes[0])

sns.distplot(cleaned_train_OL['TotalBsmtSF'], kde=True, color="green", ax=axes[1])

axes[0].set_title("Without Removing Outliers", size= 20)

axes[1].set_title("After Removing Outliers", size= 20)

plt.setp(axes, yticks=[])

plt.tight_layout()
fig, (ax1, ax2) = plt.subplots(figsize = (20,10), ncols=2, sharey=False)



sns.scatterplot(cleaned_train['TotalBsmtSF'], cleaned_train['SalePrice'],  ax=ax1)

sns.regplot(cleaned_train['TotalBsmtSF'], cleaned_train['SalePrice'], color ='m',ax=ax1)

ax1.set_title("Without Removing Outliers", size = 20)



sns.scatterplot(cleaned_train_OL['TotalBsmtSF'],cleaned_train_OL['SalePrice'], ax=ax2)

sns.regplot(cleaned_train_OL['TotalBsmtSF'],cleaned_train_OL['SalePrice'], color ='g',ax=ax2)

ax2.set_title("After Removing Outliers", size = 20)
cleaned_train['FullBath'].unique()
fig, (ax1, ax2) = plt.subplots(figsize = (20,20), nrows=2, sharey=False)



sns.swarmplot(cleaned_train['FullBath'], cleaned_train['SalePrice'],  palette="Set2",ax=ax1)

ax1.set_title("Without Removing Outliers", size = 20)



sns.swarmplot(cleaned_train_OL['FullBath'],cleaned_train_OL['SalePrice'], palette="Set2",ax=ax2)

ax2.set_title("After Removing Outliers", size = 20)
cleaned_train['TotRmsAbvGrd'].unique()
fig, (ax1, ax2) = plt.subplots(figsize = (20,20), nrows=2, sharey=False)



sns.swarmplot(cleaned_train['TotRmsAbvGrd'], cleaned_train['SalePrice'],  palette="Set2",ax=ax1)

ax1.set_title("Without Removing Outliers", size = 20)



sns.swarmplot(cleaned_train_OL['TotRmsAbvGrd'],cleaned_train_OL['SalePrice'], palette="Set2",ax=ax2)

ax2.set_title("After Removing Outliers", size = 20)
cleaned_train['YearBuilt'].head()
sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True)

sns.despine(left=True)

sns.distplot(cleaned_train['YearBuilt'], kde=True, color="m", ax=axes[0])

sns.distplot(cleaned_train_OL['YearBuilt'], kde=True, color="green", ax=axes[1])

axes[0].set_title("Without Removing Outliers", size= 20)

axes[1].set_title("After Removing Outliers", size= 20)

plt.setp(axes, yticks=[])

plt.tight_layout()
fig, (ax1, ax2) = plt.subplots(figsize = (20,10), ncols=2, sharey=False)



sns.scatterplot(cleaned_train['YearBuilt'], cleaned_train['SalePrice'],  ax=ax1)

sns.regplot(cleaned_train['YearBuilt'], cleaned_train['SalePrice'], color ='m',ax=ax1)

ax1.set_title("Without Removing Outliers", size = 20)



sns.scatterplot(cleaned_train_OL['YearBuilt'],cleaned_train_OL['SalePrice'], ax=ax2)

sns.regplot(cleaned_train_OL['YearBuilt'],cleaned_train_OL['SalePrice'], color ='g',ax=ax2)

ax2.set_title("After Removing Outliers", size = 20)
cleaned_train['Foundation'].unique()
fig, (ax1, ax2) = plt.subplots(figsize = (20,20), nrows=2, sharey=False)



sns.swarmplot(cleaned_train['Foundation'], cleaned_train['SalePrice'],  palette="Set2",ax=ax1)

ax1.set_title("Without Removing Outliers", size = 20)



sns.swarmplot(cleaned_train_OL['Foundation'],cleaned_train_OL['SalePrice'], palette="Set2",ax=ax2)

ax2.set_title("After Removing Outliers", size = 20)
y_OL = cleaned_train_OL['SalePrice']

X_OL = cleaned_train_OL.drop(['SalePrice'], axis = 1)
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()



linear_model.fit(X, y)



# Returning the R^2 for the model

cleaned_data_r2 = linear_model.score(X, y)

print('R^2: {0}'.format(cleaned_data_r2))
linear_model_OL = LinearRegression()



linear_model_OL.fit(X_OL, y_OL)



# Returning the R^2 for the model

cleaned_data_r2_OL = linear_model_OL.score(X_OL, y_OL)

print('R^2: {0}'.format(cleaned_data_r2_OL))
def calculate_residuals(model, features, label):

   

    predictions = model.predict(features)

    df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})

    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])

    

    return df_results
def linear_assumption(model, features, label):

    df_results = calculate_residuals(model, features, label)

    sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False, height=7)

    line_coords = np.arange(df_results.min().min(), df_results.max().max())

    plt.plot(line_coords, line_coords,  # X and y points

             color='darkorange', linestyle='--')

    plt.title('Actual vs. Predicted')

    plt.show()
linear_assumption(linear_model, X, y)
linear_assumption(linear_model_OL, X_OL, y_OL)
def normal_errors_assumption(model, features, label):

    df_results = calculate_residuals(model, features, label)

    plt.subplots(figsize=(12, 6))

    plt.title('Distribution of Residuals')

    sns.distplot(df_results['Residuals'])

    plt.show()
normal_errors_assumption(linear_model, X, y)
import statsmodels.api as sm

mod_fit = sm.OLS(y,X).fit()

res = mod_fit.resid

fig = sm.qqplot(res, fit = True, line = '45')

plt.show()
normal_errors_assumption(linear_model_OL, X_OL, y_OL)
import statsmodels.api as sm

mod_fit = sm.OLS(y_OL,X_OL).fit()

res = mod_fit.resid

fig = sm.qqplot(res, fit = True, line = '45')

plt.show()
sns.set_style('whitegrid')

plt.subplots(figsize = (40,30))



mask = np.zeros_like(cleaned_train.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True





sns.heatmap(cleaned_train.corr(), 

            cmap=sns.diverging_palette(20, 220, n=200), 

            mask = mask, 

            annot=True, 

            center = 0, 

           )

plt.title("Heatmap of all the Features", fontsize = 30)
from sklearn.model_selection import train_test_split

from yellowbrick.regressor import ResidualsPlot

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



model = LinearRegression()

visualizer = ResidualsPlot(model)



visualizer.fit(X_train, y_train)  

visualizer.score(X_test, y_test) 

visualizer.show()             
X_train_OL, X_test_OL, y_train_OL, y_test_OL = train_test_split(X_OL, y_OL, test_size=0.2, random_state=42)



model_OL = LinearRegression()

visualizer = ResidualsPlot(model_OL)



visualizer.fit(X_train_OL, y_train_OL)  # Fit the training data to the visualizer

visualizer.score(X_test_OL, y_test_OL)  # Evaluate the model on the test data

visualizer.show()                 # Finalize and render the figure
log_saleprice_OL = np.log1p(cleaned_train_OL["SalePrice"]) 

log_saleprice = np.log1p(cleaned_train["SalePrice"])
sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=False)

sns.despine(left=True)

sns.distplot(log_saleprice, kde=True, color="m", ax=axes[0])

sns.distplot(log_saleprice_OL, kde=True, color="green", ax=axes[1])

axes[0].set_title("Without Removing Outliers", size= 20)

axes[1].set_title("After Removing Outliers", size= 20)

plt.setp(axes, yticks=[])

plt.tight_layout()
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X, log_saleprice, test_size=0.2, random_state=42)



model = LinearRegression()

visualizer = ResidualsPlot(model)



visualizer.fit(X_train_n, y_train_n)  # Fit the training data to the visualizer

visualizer.score(X_test_n, y_test_n)  # Evaluate the model on the test data

visualizer.show()                 # Finalize and render the figure
X_train_n_OL, X_test_n_OL, y_train_n_OL, y_test_n_OL = train_test_split(X_OL, log_saleprice_OL, 

                                                                        test_size=0.2, random_state=42)



model = LinearRegression()

visualizer = ResidualsPlot(model)



visualizer.fit(X_train_n_OL, y_train_n_OL)  

visualizer.score(X_test_n_OL, y_test_n_OL)  

visualizer.show()              
def model_scores(x_train, y_train, x_test, y_test):

    lr = LinearRegression().fit(x_train, y_train)

    print('LinearRegression R^2: {0}'.format(lr.score(x_test, y_test)))

    ridge = Ridge().fit(x_train, y_train)

    print('Ridge Regression R^2: {0}'.format(ridge.score(x_test, y_test)))

    lasso = Lasso().fit(x_train, y_train)

    print('Lasso Regression R^2: {0}'.format(lasso.score(x_test, y_test)))

    rf = RandomForestRegressor().fit(x_train, y_train)

    print('RF Regression R^2: {0}'.format(rf.score(x_test, y_test)))
from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_log_error
print("R^2's for Original Dataset")

model_scores(X_train, y_train, X_test, y_test)
print("R^2's for Log of Original Dataset")

model_scores(X_train_n, y_train_n, X_test_n, y_test_n)
print("R^2's for Original Dataset After removing Outliers")

model_scores(X_train_OL, y_train_OL, X_test_OL, y_test_OL)
print("R^2's for Log of Original Dataset After removing Outliers")

model_scores(X_train_n_OL, y_train_n_OL, X_test_n_OL, y_test_n_OL)
lr = LinearRegression().fit(X_train_n_OL, y_train_n_OL)

pred1 = lr.predict(X_test_n_OL)

print('LinearRegression msle: {0}'.format(np.sqrt(mean_squared_log_error(y_test_n_OL, pred1))))
ridge = Ridge().fit(X_train_n_OL, y_train_n_OL)

pred2 = ridge.predict(X_test_n_OL)

print('Ridge Regression msle: {0}'.format(np.sqrt(mean_squared_log_error(y_test_n_OL, pred2))))
lasso = Lasso().fit(X_train_n_OL, y_train_n_OL)

pred3 = lasso.predict(X_test_n_OL)

print('Lasso Regression msle: {0}'.format(np.sqrt(mean_squared_log_error(y_test_n_OL, pred3))))
rf = RandomForestRegressor().fit(X_train_n_OL, y_train_n_OL)

pred3 = rf.predict(X_test_n_OL)

print('RF Regression msle: {0}'.format(np.sqrt(mean_squared_log_error(y_test_n_OL, pred3))))
def msle_score_ridge(alpha):

    ridge = Ridge(alpha).fit(X_train_n_OL, y_train_n_OL)

    pred2 = ridge.predict(X_test_n_OL)

    score_ridge = np.sqrt(mean_squared_log_error(y_test_n_OL, pred2))

    return(score_ridge)
scores_ridge = {}

for i in [0.001, 0.01, 0.1, 1, 10, 100]:

    scores_ridge[i] = msle_score_ridge(i)
scores_ridge
ridge = Ridge(alpha = 10).fit(X_train_n_OL, y_train_n_OL)

pred2 = ridge.predict(X_test_n_OL)

score_ridge = np.sqrt(mean_squared_log_error(y_test_n_OL, pred2))

print(score_ridge)
pred = ridge.predict(cleaned_test)

back  = np.e**pred
output = pd.DataFrame({'Id': test_Id,

                       'SalePrice': back})

output.to_csv('submission.csv', index=False)