# Import the relvant libraries etc

# Initially I will be doing th exploratory data analysis so will only import numpy,panda and matplotlib



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # import matplotlib for plotting

from pandas import DataFrame,Series

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Import the House Price training dataset

HPriceData = DataFrame(pd.read_csv('../input/train.csv'))

test = DataFrame(pd.read_csv('../input/train.csv'))
# Take a quick look at the data which has been imported 

HPriceData.head()
HPriceData.columns
print('HPriceData shape is', HPriceData.shape)

print('test shape is', test.shape)
from types import *

# Create a for loop to loop through each column and output:

# - the type of data held within it

# - the number of missing values

# - the correlation between it and the sale price



# Create a new DataFrame to store the info we will generate

HPriceInfo = DataFrame(columns=['Variable','Missing_data','Correlation'])

CatVars = []

for column in HPriceData.columns:

    List = []

    List.append(column)

    # List.append(HPriceData[column].dtype)

    List.append(HPriceData[column].isnull().sum())

    if (HPriceData[column].dtype == 'int64') or (HPriceData[column].dtype == 'float64') :

        List.append(HPriceData[column].corr(HPriceData['SalePrice']))

    else:

        List.append(np.nan)

        #print(HPriceData[column].dtype)

        CatVars.append(column)



    print(List)

    HPriceInfo = HPriceInfo.append(pd.Series(List[:], index=['Variable','Missing_data','Correlation']), ignore_index=True)

    # HPriceInfo.append(List)

HPriceInfo  

print(CatVars)
print(CatVars)

len(CatVars)
# if the correlation  

HPriceInfo.sort_values(by=['Correlation'], ascending=False)

# Any column with a correlation of above 0.5 is likely to have a significant impact on the house price

# Lets create a new list which only contains the names of the columns which satisfy this criteria

HPriceInfoNewIndex = HPriceInfo.set_index('Variable')

Usefull = []

NonNumeric = []

for var in range(len(HPriceInfoNewIndex)):

    print(var)

    

    if HPriceInfoNewIndex.iloc[var]['Correlation'] > 0.5:

        Usefull.append(HPriceInfo.iloc[var]['Variable'])



    elif (pd.isna(HPriceInfoNewIndex.iloc[var]['Correlation']) == True and HPriceInfoNewIndex.iloc[var]['Missing_data'] < 100):

        NonNumeric.append(HPriceInfo.iloc[var]['Variable'])

       

print(Usefull)    

print(NonNumeric)

    
%matplotlib inline



a = 1

plt.figure(figsize=(20,40))

y = HPriceData['SalePrice']

for i in Usefull:

    x = HPriceData[i]

    plt.subplot(6,2,a).scatter(x,y)

    a += 1

# Iterate through the list of the non-numeric variables, making a list of the different values included

for i in NonNumeric: 

    a = Series(HPriceData[i])

    a = a.unique()

    for j in a:

        b =HPriceData.loc[HPriceData[i] == j]

        #print(b)

    #print(a)

    

    

print(len(NonNumeric))
%matplotlib inline



a = 1

plt.figure(figsize=(20,100))

y = HPriceData['SalePrice']

for i in NonNumeric:

    x = HPriceData[i].fillna(value='Na')

    plt.subplot(10,4,a).scatter(x,y)

    a += 1
HPriceData.shape
# First we need to either remove missing data or set it to a mean value



# Loop through each column, find the number of missing values in the column,

# it it is great than 20% drop that column

for column in HPriceData.columns:

    print(str(column) + 'has ' + str(HPriceData[column].isnull().sum()) + ' missing values')

    if HPriceData[column].isnull().sum() > (1460/5):

        HPriceData = HPriceData.drop(columns=[column])

        print('dropped ' + str(column))

print(HPriceData.shape)

        
CatVars
Removals = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']

for item in Removals:

    if item in CatVars: 

        CatVars.remove(item)

        print('removed ' + item)

        
DummyVars = pd.get_dummies(HPriceData, prefix=CatVars, prefix_sep="__",

                              columns=CatVars)
DummyVars.head()

DummyVars.shape

test = DummyVars
for column in test.columns:

    print(str(column) + 'has ' + str(test[column].isnull().sum()) + ' missing values')

    if test[column].isnull().sum() > (1460/5):

        test = tes.drop(columns=[column])

        print('dropped ' + str(column))

#print(test.shape)
for column in test.columns:

    #print(str(column) + 'has ' + str(test[column].isnull().sum()) + ' missing values')

    test[column].fillna(test[column].mean(),inplace=True)

    #print(test[column].isnull().sum())

train_manipulated = test
test = DataFrame(pd.read_csv('../input/test.csv'))

for column in test.columns:

    #print(str(column) + 'has ' + str(test[column].isnull().sum()) + ' missing values')

    if test[column].isnull().sum() > (1460/5):

        test = test.drop(columns=[column])

        #print('dropped ' + str(column))

print(test.shape)

        
test = pd.get_dummies(test, prefix=CatVars, prefix_sep="__",

                              columns=CatVars)
test.head()
test.shape

test.columns
test_cols = []

for i in test.columns:

    test_cols.append(i)



train_cols = []

for j in train_manipulated.columns:

    train_cols.append(j)

    
print(len(test_cols))

print(len(train_cols))



for item in train_cols:

    if item not in test_cols: 

        print('test ' +item)

for item in test_cols:

    if item not in train_cols: 

        print('trian ' + t_cols)
X = train_manipulated.drop(columns=['SalePrice'])

X = X[test_cols]

print(X.shape)

y = train_manipulated.SalePrice

from sklearn.linear_model import LinearRegression

Train_fit = LinearRegression().fit(X,y)
Train_fit.score(X,y)
from sklearn.ensemble import RandomForestRegressor

Train_fit_Forest = RandomForestRegressor(n_estimators=1000)

Train_fit_Forest.fit(X,y)
Train_fit_Forest.score(X,y)
for column in test.columns:

    print(str(column) + 'has ' + str(test[column].isnull().sum()) + ' missing values')

    test[column].fillna(test[column].mean(),inplace=True)

    print(test[column].isnull().sum())
predicted_prices = Train_fit_Forest.predict(test)
train_manipulated.shape
TomBale_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here

TomBale_submission.to_csv('TomBale_submission.csv', index=False)

TomBale_submission.head()