import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.display.max_columns = None    # to show all the features
pd.options.display.max_rows = 90         # to show 90 observation
from IPython.core.interactiveshell import InteractiveShell # to show multiple output of a cell
InteractiveShell.ast_node_interactivity = "all"
data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data.shape
data.sample(5)
data.info()
data.describe()
data.isnull().sum()
# Percentage of missing data in each feature

for col_names in data.columns:
    msng = (data[col_names].isnull().sum())/len(data)
    print('{} - {}%'.format(col_names, msng*100))
# Droping features with more than 80% msng data or insignificant feature

data.drop(['MiscFeature','Fence','PoolQC','Alley','Id'], axis=1, inplace=True)
# non-numerical features

non_num_feat = data.select_dtypes(exclude=(np.number)).columns.values
non_num_feat
# Imputing all non-numerical features with mode

for colmn_names in data[non_num_feat].columns:
    msing = data[colmn_names].isnull().sum()
    
    if msing > 0:
        print('Imputing missing value for {} with mode.'.format(colmn_names))
        data[colmn_names].fillna(data[colmn_names].mode()[0],inplace=True)
        
    else:
        print('No value is missing for {}.'.format(colmn_names))
num_feat = data.select_dtypes(include=(np.number)).columns.values
num_feat
# Imputing numerical features with mean

for column_names in data[num_feat].columns:
    num_msng = data[column_names].isnull().sum()
    
    if num_msng>0:
        print('Imputing missing value of {} with mean'.format(column_names))
        data[column_names].fillna(round(data[column_names].mean()),inplace=True)
        
    else:
        print('No value is missing for {}.'.format(column_names))
data.isnull().sum() # no data is missing now.
data.drop_duplicates(ignore_index=True) # No data is duplicating
from sklearn.model_selection import train_test_split
train,test = train_test_split(data, test_size=0.2, random_state=10, shuffle=True)
data.shape
train.shape
test.shape
X_train = train.drop(['SalePrice'], axis=1)  # train independent var
Y_train = train['SalePrice']                 # train dependent var
X_test = test.drop(['SalePrice'], axis=1)    # test independent var
Y_test = test['SalePrice']                   # test dependent var    
X_train.shape
Y_train.shape
X_test.shape
Y_test.shape
plt.figure(figsize=(40,20))
sns.heatmap(data.corr(), annot=True)
sns.set(font_scale=1)
data['SalePrice'].describe()
plt.figure(figsize=(20,10))
sns.boxplot(x='RoofStyle',y='SalePrice', data=data)
sns.set(font_scale=1.5)
plt.figure(figsize=(20,10))
sns.lineplot(x='MSSubClass',y='SalePrice',data=data, ci=None)
sns.set(font_scale=1.5)
plt.figure(figsize=(20,10))
sns.boxenplot(x='MSZoning', y='SalePrice', data=data)
sns.set(font_scale=1.5)
plt.figure(figsize=(20,10))
sns.scatterplot(x='LotFrontage',y='SalePrice',data=data)
sns.set(font_scale=1.5)
plt.figure(figsize=(20,10))
sns.boxplot(x='Street',y='SalePrice',data=data, hue='RoofStyle')
sns.set(font_scale=1.5)
plt.figure(figsize=(20,10))
sns.scatterplot(x='LotArea',y='SalePrice',data=data)
sns.set(font_scale=1.5)
plt.figure(figsize=(20,10))
sns.boxenplot(x='LotConfig',y='SalePrice',data=data)
sns.set(font_scale=1.5)
plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
sns.barplot(x='YearBuilt',y='SalePrice',data=data)
sns.set(font_scale=1)
plt.figure(figsize=(20,10))
sns.boxplot(x='LotShape',y='SalePrice',data=data)
sns.set(font_scale=1.5)
sns.lmplot(x='MasVnrArea',y='SalePrice',data=data,height=8,aspect=2,fit_reg=True,scatter_kws={'s':10})
sns.set(font_scale=1.5)
plt.figure(figsize=(20,10))
plt.xticks(rotation=45)
sns.boxplot(x='Neighborhood',y='SalePrice',data=data)
sns.set(font_scale=1)
sns.lmplot(x='BsmtFinSF1',y='SalePrice',data=data,height=8,aspect=2, scatter_kws={'s':10})
sns.set(font_scale=1.5)
plt.figure(figsize=(20,10))
sns.boxenplot(x='BsmtFinType2',y='SalePrice',data=data)
sns.set(font_scale=1.5)
sns.lmplot(x='BsmtUnfSF',y='SalePrice',data=data,height=8,aspect=2, scatter_kws={'s':10})
sns.set(font_scale=1.5)
plt.figure(figsize=(20,10))
sns.boxplot(x='OverallQual', y='SalePrice', data=data)
sns.set(font_scale=1.5)
X_train = pd.get_dummies(X_train,drop_first=True)
X_train.sample(10)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,Y_train)
lin_reg.intercept_
lin_reg.coef_
train_pred = lin_reg.predict(X_train)
train_pred
rmse = np.sqrt(sum((train_pred - Y_train)**2)/X_train.shape[0])
print('RMSE obtained for train is {}'.format(rmse))
# encodint test data

X_test = pd.get_dummies(X_test, drop_first=True)
X_test.sample(10)
X_test.shape
X_train.shape
msn_test_col = set(X_train.columns) - set(X_test.columns) 


for col in msn_test_col:
    X_test[col] = 0
    
X_test = X_test[X_train.columns]
X_test.shape
X_train.shape
test_pred = lin_reg.predict(X_test)
test_pred
rmse_test = np.sqrt(sum((test_pred - Y_test)**2)/X_test.shape[0])
print('RMSE obtained for test is {}'.format(rmse_test))