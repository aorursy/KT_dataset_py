# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None) 

sns.set_palette("Set2")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

sample = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")

text_file = open("/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt", "r")

description = text_file.read()
train.shape
train.info()
objectscol = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',

       'SaleType', 'SaleCondition']  #<---train.describe(include =np.object).columns



infotable = train.describe(include =np.object)

infotable
for attribute in objectscol:

    x = infotable.loc['count'][attribute]/1460

    if x < 0.1:print(attribute)
train = train.drop(["Alley","PoolQC","MiscFeature"], axis = 1)

train.shape
objectcol = train.describe(include = np.object).columns

train[objectcol] = train[objectcol].fillna("missing")
sns.countplot(train["Fence"])
len(objectcol)
fig = plt.figure(figsize=(10,50))

fig.subplots_adjust(hspace=0.8, wspace=0.8)



for i in range(1, 41):

    ax = fig.add_subplot(20, 2, i)

    g = sns.countplot(train[objectcol[i-1]])

    plt.xticks(rotation=45)

    
flotcol = train.describe(include = np.float64).columns

infortable2 = train.describe(include = np.float64)

for attribute in flotcol:

    x = infortable2.loc['count'][attribute]/1460

    if x < 0.1:

        print("found 90% null attribute:{}",attribute)

    else:

        print("no attribute found with over 90% empty")

        pass

        #nothing happend

train[flotcol]=train[flotcol].fillna(0)

fig = plt.figure(figsize=(15,5))

fig.subplots_adjust(hspace=0.8, wspace=0.8)

for i in range(1, 4):

    ax = fig.add_subplot(1, 3, i)

    sns.distplot(train[flotcol[i-1]])
# INT64 attribiutes

intcol = train.describe(include = np.int64).columns

infotable3 = train.describe(include = np.int64)

infotable3
for attribute in intcol:

    x = infotable3.loc['count'][attribute]/1460

    if x < 0.1:

        print("found 90% null attribute:{}",attribute)

    else:

        pass

        #nothing happend

print("finished")
len(intcol)
#sns.distplot(train["MSSubClass"])

fig = plt.figure(figsize=(15,15))

fig.subplots_adjust(hspace=0.8, wspace=0.8)

for i in range(1, 36):

    ax = fig.add_subplot(7, 5, i)

    sns.distplot(train[intcol[i-1]])



sns.violinplot(y = "SalePrice", x = "YrSold", data = train)
from sklearn.ensemble import ExtraTreesRegressor

target = train['SalePrice']

features = train.drop(columns = ['SalePrice','Id'])

#only possible for numeric features, so we here only use numeric

non_obj_features = features.describe(exclude = np.object).columns

features = features[non_obj_features]

beforefs = features.columns

print("before feature selection:{} number of features.".format(len(non_obj_features)))

#create a classifer for feature selection

forest  = ExtraTreesRegressor(n_estimators = 50)

forest = forest.fit(features, target)

importance_table = pd.Series(forest.feature_importances_, index=features.columns)

importance_table = importance_table.sort_values(ascending = False)



plt.figure(figsize=(16,8))

ax = sns.barplot(x = importance_table.index[:10], y = importance_table.values[:10], palette="BuGn_r")

ax.set_title('Feature Importance')



select_feature = importance_table.index[:10]

#importance_table
select_feature 
fig = sns.PairGrid(train[select_feature[0:5]])

fig.map_offdiag(plt.scatter)

fig.map_diag(sns.distplot, bins=30)
fig = sns.PairGrid(train[select_feature[5:]])

fig.map_offdiag(plt.scatter)

fig.map_diag(sns.distplot, bins=30);
fig = plt.figure(figsize=(10,20))

fig.subplots_adjust(hspace=0.5, wspace=0.5)



for i in range(1, 11):

    ax = fig.add_subplot(5, 2, i)

    g = sns.scatterplot(x = train[select_feature[i-1]], y="SalePrice", data=train)

    g.set_xlabel(g.get_xlabel(),fontsize= 20)

    g.set_ylabel(g.get_ylabel(),fontsize= 15)

    plt.xticks(rotation=45)
fig = plt.figure(figsize=(20,150))

fig.subplots_adjust(hspace=0.5, wspace=0.5)



for i in range(1, 41):

    ax = fig.add_subplot(20, 2, i)

    g = sns.violinplot(x = train[objectcol[i-1]], y="SalePrice", data=train)

    g.set_xlabel(g.get_xlabel(),fontsize= 20)

    g.set_ylabel(g.get_ylabel(),fontsize= 15)

    plt.xticks(rotation=45)
len(objectcol)
test[select_feature].info()
#Checking misisng valus

missing_val_count_by_column = (test[select_feature].isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
test[select_feature].info()
missing_val_count_by_column = (test[select_feature].isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
missing_cols =missing_val_count_by_column[test[select_feature].isnull().sum()>0].index

for i in range(0,len(missing_cols)):   

    fillwith = test[missing_cols[i]].mean()

    test[missing_cols[i]].fillna(value=fillwith, inplace  = True)
X = train.drop(columns =['Id','SalePrice'])
print("There are {} features in used. ".format(len(X.columns)))
X = pd.get_dummies(X)
print("After dummies applied, There are {} features in used. ".format(len(X.columns)))
X_test = test.drop(columns =['Id'])

X_test = pd.get_dummies(X_test)
missing_val_count_by_column = (X_test.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])

#missing_cols =missing_val_count_by_column[X_test.isnull().sum()>0].index

for i in range(0,len(missing_cols)):   

    fillwith = X_test[missing_cols[i]].mean()

    X_test[missing_cols[i]].fillna(value=fillwith, inplace  = True)
X_test =X_test.drop(columns =['PoolQC_Gd', 'Alley_Pave', 'Alley_Grvl', 'MiscFeature_Shed', 'MiscFeature_Othr', 'PoolQC_Ex', 'MiscFeature_Gar2'])
X = X[X_test.columns]
missing_val_count_by_column = (X_test.isnull().sum())

missing = missing_val_count_by_column[missing_val_count_by_column > 0]

missing_cols =missing_val_count_by_column[X_test.isnull().sum()>0].index

for i in range(0,len(missing)):   

    fillwith = X_test[missing_cols[i]].mean()

    print(fillwith)

    X_test[missing_cols[i]] = X_test[missing_cols[i]].fillna(value=fillwith)
# from sklearn.ensemble import RandomForestRegressor

# forest = RandomForestRegressor(n_estimators=100)

# forest.fit(X,target)

# result = forest.predict(X_test)

# handin = pd.DataFrame({'Id': test.Id,'SalePrice': result})

# handin.to_csv('submission.csv', index=False)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV





parameters  = { 

    'n_estimators': [50,100,200,300,700],

    'max_features': ['auto', 'sqrt', 'log2']

}

forest = RandomForestRegressor()

#apply grid search cv method on the forest and parameters.

gs_forest = GridSearchCV(forest, parameters, cv = 5)

gs_forest.fit(X, target)
print("Best score: {}, \nBest params: {}".format(gs_forest.best_score_,gs_forest.best_estimator_))
#Call predict on the estimator with the best found parameters.

result = gs_forest.predict(X_test)

handin = pd.DataFrame({'Id': test.Id,'SalePrice': result})

handin.to_csv('submission.csv', index=False)
# from sklearn.preprocessing import StandardScaler

# from sklearn.decomposition import PCA

# from sklearn.svm import SVR



# scaler = StandardScaler()

# scaler.fit(X)

# X = scaler.transform(X)

# X_test = scaler.transform(X_test)

# pca = PCA(n_components=10)

# pca.fit(X)

# X = pca.transform(X)

# X_test = pca.transform(X_test)

# clf = SVR(gamma='scale', C=1.0, epsilon=0.2)

# clf.fit(X,target)

# result = clf.predict(X_test)

# handin = pd.DataFrame({'Id': test.Id,'SalePrice': result})

# handin.to_csv('submission.csv', index=False)

# print("score:",0.4)