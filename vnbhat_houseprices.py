# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import numpy as np

import pandas as pd

import re

from sklearn import datasets, linear_model

import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter

import tensorflow as tf

from tensorflow import keras

import seaborn as sns

import scipy.stats as stats

from collections import Counter

from sklearn.model_selection import KFold

from sklearn import preprocessing

pd.set_option("display.latex.repr", True)

import xgboost as xgb

# Any results you write to the current directory are saved as output.
train_set = pd.read_csv('../input/train.csv')

test_set = pd.read_csv('../input/test.csv')

ids = test_set['Id']

train_set.head(5)
print ("\n\n---------------------")

print ("TRAIN SET INFORMATION")

print ("---------------------")

print ("Shape of training set:", train_set.shape, "\n")

print ("Column Headers:", list(train_set.columns.values), "\n")

#print (train_set.describe(), "\n\n")

#print (train_set.dtypes)
print ("\n\n--------------------")

print ("TEST SET INFORMATION")

print ("--------------------")

print ("Shape of test set:", test_set.shape, "\n")

print ("Column Headers:", list(test_set.columns.values), "\n")

#print (test_set.describe(), "\n\n")

#print (test_set.dtypes)
missing_values = []

nonumeric_values = []



print ("TRAINING SET INFORMATION")

print ("========================\n")

missing_values_set_train = []

for column in train_set:

    # Find all the unique feature values

    uniq = train_set[column].unique()

    print ("'{}' has {} unique values" .format(column,uniq.size))

    if (uniq.size > 25):

        print("~~Listing up to 25 unique values~~")

    print (uniq[0:24])

    print ("\n-----------------------------------------------------------------------\n")



    # Find features with missing values

    if (True in pd.isnull(uniq)):

        s = "{} has {} missing" .format(column, pd.isnull(train_set[column]).sum())

        missing_values_set_train.append(column)

        missing_values.append(s)

        

    # Find features with non-numeric values

    for i in range (1, np.prod(uniq.shape)):

        if (re.match('nan', str(uniq[i]))):

            break

        if not (re.search('(^\d+\.?\d*$)|(^\d*\.?\d+$)', str(uniq[i]))):

            nonumeric_values.append(column)

            break

nonumeric_values_train = nonumeric_values

missing_values_train = missing_values

print ("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

#print ("Features with missing values:\n{}\n\n" .format(missing_values))

#print ("Features with non-numeric values:\n{}" .format(nonumeric_values))

print ("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
#sns.pairplot(train_df[col], size=3);

plt.figure()

otl = sns.lmplot('GrLivArea', 'SalePrice',data=train_set, fit_reg=False);
train_set.drop(train_set[(train_set['GrLivArea'] > 4000)].index,inplace=True)



plt.figure()

sns.lmplot('GrLivArea', 'SalePrice',data=train_set, fit_reg=False);

plt.xlim(0,5500);

plt.ylim(0,800000);
sns.distplot(train_set['SalePrice'])

plt.title('SalePrice Distribution')

plt.ylabel('Frequency')



plt.figure()

qq = stats.probplot(train_set['SalePrice'], plot=plt)

plt.show()



# For normally distributed data, the skewness should be about zero. 

# A skenewss  value greater than zero means that there is more weight in the left tail of the distribution



print("Skewness: {:.3f}".format(train_set['SalePrice'].skew()))
plt.figure(figsize=(25,5))



# correlation table

corr_train = train_set.corr()



# select top 10 highly correlated variables with SalePrice

num = 10

col = corr_train.nlargest(num, 'SalePrice')['SalePrice'].index

coeff = np.corrcoef(train_set[col].values.T)



# heatmap

heatmp = sns.heatmap(coeff, annot = True, xticklabels = col.values, yticklabels = col.values, linewidth=2,cmap='PiYG', linecolor='blue')
sns.pairplot(train_set[col], size=3);
df = train_set

mc = pd.DataFrame(df.isnull().sum(),columns=['Missing Count'])

mc = mc[mc['Missing Count']!=0]

mc['Missing %'] = (mc['Missing Count'] / df.shape[0]) * 100

mc.sort_values('Missing %',ascending=False)
# seperate the target variable (SalePrice) from the train



y_df = train_set['SalePrice']

train_set.drop('SalePrice',axis=1,inplace=True)



print('dimension of the train:' , train_set.shape)

print('dimension of the test:' , test_set.shape)
# In order to avoid repeating unnecessary codes, for our convenience, let's combine the train and test set.

df = pd.concat([train_set, test_set]).reset_index()



df.drop(['index'],axis=1,inplace=True)
print('dimension of the dataset:' , df.shape)

df.head()
mc = pd.DataFrame(df.isnull().sum(),columns=['Missing Count'])

mc = mc[mc['Missing Count']!=0]

mc['Missing %'] = (mc['Missing Count'] / df.shape[0]) * 100

mc.sort_values('Missing %',ascending=False)
nones = ['PoolQC', 'MiscFeature', 'Alley','Fence', 'FireplaceQu', 'GarageType','GarageFinish',

        'GarageQual','GarageCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

        'MasVnrType']



for none in nones:

    df[none].fillna('None',inplace = True)
zeros = ['GarageYrBlt','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',

         'BsmtFullBath','BsmtHalfBath','MasVnrArea']



for zero in zeros:

    df[zero].fillna(0, inplace = True)
Counter(df.Utilities)

df.drop('Utilities',axis=1, inplace=True)
freq = ['MSZoning','Exterior1st','Exterior2nd','SaleType','Electrical','KitchenQual','Functional']



for fr in freq:

    df[fr].fillna(df[fr].mode()[0], inplace=True)
df['old_lotfrontage'] = df['LotFrontage']



df['LotFrontage'] = df.groupby(['LotArea','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,5))

ol = sns.distplot(df['old_lotfrontage'].dropna(),ax=ax1,kde=True,bins=70)

lf = sns.distplot(df['LotFrontage'],ax=ax2,kde=True,bins=70,color='red')



# drop the old_lotfrontage as we finished the comparison

df.drop('old_lotfrontage',axis=1,inplace=True)
# get_dummies can convert data to 0 and 1 only if the data type is string. Among the many nominal features,

# MSSubClass, MoSold, and YrSold are integer type so we need to convert them to string type.



df['MoSold'] = df.astype(str)

df['YrSold'] = df.astype(str)

df['MSSubClass'] = df.astype(str)



nominals = ['MSSubClass','MSZoning','Street','Alley','LandContour','LotConfig','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',

           'Exterior1st','Exterior2nd','MasVnrType','Foundation','Heating','CentralAir','GarageType','MiscFeature','SaleType','SaleCondition','MoSold','YrSold']
from sklearn.preprocessing import LabelEncoder



ordinals = ['LotShape','LandSlope','OverallQual','OverallCond','ExterQual','ExterCond','BsmtQual',

           'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','Electrical','KitchenQual',

            'Functional','FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence']



for ordinal in ordinals:

    le = LabelEncoder()

    le.fit(df[ordinal])

    df[ordinal] = le.transform(df[ordinal])
# Total square feet of houses



df['totalArea'] = df['GrLivArea'] + df['TotalBsmtSF']
# Assign numeric features by excluding non numeric features

numeric = df.dtypes[df.dtypes != 'object'].index



# Display the skewness of each column and sort the values in descending order 

skewness = df[numeric].apply(lambda x: x.skew()).sort_values(ascending=False)



# Create a dataframe and show 5 most skewed features 

sk_df = pd.DataFrame(skewness,columns=['skewness'])

sk_df['skw'] = abs(sk_df)

sk_df.sort_values('skw',ascending=False).drop('skw',axis=1).head()
train_set.drop('PoolQC',axis=1,inplace = True)

train_set.drop('MiscVal',axis=1,inplace = True)

df = pd.get_dummies(df)

print(df.shape)
from sklearn.tree import DecisionTreeRegressor
# Split the combined dataset into two: train and test

kf = KFold(n_splits=3)

kf.get_n_splits(train_set)



from sklearn.model_selection import KFold

import xgboost

#added some parameters

kf = KFold(n_splits = 5, shuffle = True, random_state = 2)

models = []

for i in range(5):

    result = next(kf.split(df[:train_set.shape[0]]), None)

    X_train = df[:train_set.shape[0]].iloc[result[0]]

    X_test =  df[:train_set.shape[0]].iloc[result[1]]

    y_train = y_df.iloc[result[0]]

    y_test = y_df.iloc[result[1]]

    train_id = X_train.iloc[:,0]

    test_id = X_test.iloc[:,0]

    X_train.drop('Id',axis=1,inplace = True)

    X_test.drop('Id',axis=1,inplace = True)

    scaler = preprocessing.StandardScaler().fit(X_train)

    scaler.transform(X_train) 

    scaler.transform(X_test)

    

    xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,

                           colsample_bytree=1, max_depth=10)#learning_rate = 0.08

    

    xgb.fit(X_train, y_train)

    # Fit regression model - Decision Tree Regressor

    #regr_1 = DecisionTreeRegressor(max_depth=2)

    #regr_2 = DecisionTreeRegressor(max_depth=10)

    #regr_1.fit(X_train, y_train)

    #regr_2.fit(X_train, y_train)

    #models.append(regr_2)

    # Predict

    #X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]

    #y_1 = regr_1.predict(X_test)

    #y_2 = regr_2.predict(X_test)

    y_1 = xgb.predict(X_test)

    # Plot the results

    plt.figure()

    plt.scatter(y_1, y_test)

    #plt.scatter(X_train, y_train)#, s=20, edgecolor="black",c="darkorange", label="data")

    #plt.plot(X_test, y_test, color="cornflowerblue",label="max_depth=2", linewidth=2)

    #plt.plot(X_test, y_test, color="yellowgreen", label="max_depth=5", linewidth=2)

    #plt.xlabel("data")

    #plt.ylabel("target")

    #plt.title("Decision Tree Regression")

    #plt.legend()

    plt.show()
df_test = df[train_set.shape[0]:]

df_test.drop('Id',axis=1,inplace = True)

df_test.shape

test_predictions = xgb.predict(df_test)#.flatten()
submission = pd.DataFrame({

       "Id": ids,

       "SalePrice": test_predictions})
submission.to_csv('house_price 2.csv', index=False)