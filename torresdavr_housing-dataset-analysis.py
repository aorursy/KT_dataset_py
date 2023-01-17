import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import os

print(os.listdir('../input'))

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

datasets = [train_data,test_data]
pd.set_option('display.max_columns', None)  

train_data.head()
pd.set_option('display.max_columns', None)  

test_data.head()
"""

train_df.shape  (1460, 81)

test_df.shape   (1459, 80)

"""

train_data.shape
test_data.shape
col = ['SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea','LowQualFinSF','YrSold','OverallCond','MSSubClass', 'EnclosedPorch','KitchenAbvGr']   

corr_cols = train_data[col]



sns.pairplot(corr_cols);
train_data['OverallQual'].value_counts()
train_data['OverallQual'].isnull().sum()
sns.scatterplot(x='OverallQual', y='SalePrice', data=train_data);
train_data['OverallCond'].value_counts()
train_data['OverallCond'].isnull().sum()
sns.scatterplot(x='OverallCond', y='SalePrice', data=train_data);
sns.boxplot(x='OverallCond',y='SalePrice',data=train_data,palette='winter');
sns.lmplot(x='GrLivArea',y='SalePrice',data=train_data);
train_data['SalePrice'].describe()
sns.distplot(train_data['SalePrice']);
print('The SalePrice skew is: ', train_data['SalePrice'].skew())
numeric_features = train_data.select_dtypes(include=[np.number])

corr = numeric_features.corr()

print(corr['SalePrice'].sort_values(ascending=False)[:10],'\n')

print(corr['SalePrice'].sort_values(ascending=False)[-10:])
corr_price = train_data.corr()

corr_price

#corr_sale_price = corr_price['SalePrice']

#corr_sale_price
plt.figure(figsize=(10,10))

sns.heatmap(corr_cols.corr(),annot=True);
"""

train_data.shape  (1460, 81)

test_data.shape   (1459, 80)

"""

def percent_missing_values(df):

    mis_val = df.isnull().sum()

    mis_val_percent = 100 * df.isnull().sum() / len(df)

    mis_val_table = pd.concat([mis_val,mis_val_percent],axis=1)

    mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    return mis_val_table_ren_columns           
percent_miss = percent_missing_values(train_data).sort_values(by='Missing Values',ascending=False)   

percent_miss[:25]
"""

(1460, 76)

(1459, 75)

"""

null_features = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'] 

train_df = train_data.drop(null_features, axis=1)

test_df = test_data.drop(null_features,axis=1)
numeric_feat_train = train_df.select_dtypes(include=[np.number])

numeric_feat_test = test_df.select_dtypes(include=[np.number])
numeric_feat_train.head()
numeric_feat_train.isnull().sum()
num_fill_na_test = numeric_feat_test.apply(lambda x: x.fillna(x.mean())) 

num_fill_na_test.head()
num_fill_na_train = numeric_feat_train.apply(lambda x: x.fillna(x.mean())) 
num_fill_na_test.isnull().sum()
"""

num_fill_na_train

num_fill_na_test



"""

categors_train = train_df.select_dtypes(exclude=[np.number])

categors_test = test_df.select_dtypes(exclude=[np.number])
train_d_1 = pd.concat([num_fill_na_train,categors_train], axis=1)

test_d_1 = pd.concat([num_fill_na_test,categors_test], axis=1)
print(train_d_1.shape)

print(test_d_1.shape)
categors = train_d_1.select_dtypes(exclude=[np.number])
categors['BsmtQual'].value_counts()
categors['BsmtFinType1'].value_counts()
train_no_nan = train_d_1.copy()

test_no_nan = test_d_1.copy()
cols = ['GarageCond','GarageQual','GarageFinish','GarageType']





train_no_nan[cols] = train_no_nan[cols].replace({np.NaN:'No Garage'})

test_no_nan[cols] = test_no_nan[cols].replace({np.NaN:'No Garage'})
col_base = ['BsmtFinType2','BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1']





train_no_nan[col_base] = train_no_nan[col_base].replace({np.NaN:'No Basement'})

test_no_nan[col_base] = test_no_nan[col_base].replace({np.NaN:'No Basement'})
train_no_nan['GarageQual'].value_counts()
test_no_nan['GarageQual'].value_counts()
train_no_nan['BsmtFinType1'].value_counts()
print(train_no_nan.shape)

print(test_no_nan.shape)
#60% is None so I'll change NaN to None



test_no_nan['MasVnrType'].value_counts(normalize=True)
test_no_nan.loc[544]
train_no_nan['MasVnrType'] = train_no_nan['MasVnrType'].replace({np.NaN:'None'})

test_no_nan['MasVnrType'] = test_no_nan['MasVnrType'].replace({np.NaN:'None'})
test_no_nan['MasVnrType'].value_counts()
print(train_no_nan.shape)

print(test_no_nan.shape)
train_d = train_no_nan.dropna()

test_d = test_no_nan.dropna()
print(train_d.shape)

print(test_d.shape)
cat_train = train_d.select_dtypes(exclude=[np.number])

cat_test = test_d.select_dtypes(exclude=[np.number])
print(cat_train.shape)

print(cat_test.shape)
train_dummies = pd.get_dummies(cat_train, drop_first=True)
train_dummies.head()


test_dummies = pd.get_dummies(cat_test, drop_first=True)
test_dummies.head()
print(train_dummies.shape)

print(test_dummies.shape)
train_dt = pd.concat([train_d, train_dummies], axis=1)

test_dt = pd.concat([test_d, test_dummies], axis=1)
train_df.shape
print(train_dt.shape)

print(test_dt.shape)
train_dt['SaleType'].value_counts()
cat_train.columns
cat_train['LandSlope'].value_counts()
len(cat_test.columns)
cols = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',

       'PavedDrive', 'SaleType', 'SaleCondition']

train_drop = train_dt.drop(cols, axis=1)
test_drop = test_dt.drop(cols, axis=1)
print(train_drop.shape)

print(test_drop.shape)
X = train_drop.drop(['SalePrice'], axis=1)

y = np.log(train_drop['SalePrice'])
print(X.shape)

print(y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

model = lr.fit(X_train,y_train)
model.score(X_test,y_test)
y_pred = model.predict(X_test)
from sklearn import metrics
print(metrics.mean_absolute_error(y_test,y_pred))

print(metrics.mean_squared_error(y_test,y_pred))

print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
actual_values = y_test

plt.scatter(y_pred, actual_values, alpha=.75,

            color='b') #alpha helps to show overlapping data

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Linear Regression Model')
submission = pd.DataFrame({

        "PassengerId": X_test["Id"],

        "Sale Price": y_pred

    })

submission.to_csv('housing_submit.csv', index=False)
housing = pd.read_csv('housing_submit.csv')
housing