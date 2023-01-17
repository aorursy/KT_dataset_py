# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import linear_model

from sklearn import cross_validation

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')



train_df = train_df[['Id', 'LotArea', 'SalePrice', 'TotalBsmtSF', 'GrLivArea','GarageCars', 'YearBuilt', 'YrSold', 'PoolArea', 'TotRmsAbvGrd', 'FullBath', 'HalfBath', 'KitchenAbvGr', 'BedroomAbvGr', 'GarageArea','BedroomAbvGr','BsmtFullBath','BsmtHalfBath', 'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','LotFrontage','BsmtUnfSF','LowQualFinSF','BsmtFinSF1','BsmtFinSF2', 'MSZoning']]

#train_df['TotalBuildArea'] = train_df['GrLivArea'] + train_df['TotalBsmtSF']

#train_df['TotalBathAbvGr'] = train_df['FullBath'] + train_df['HalfBath']

train_df['NumOfYear'] = train_df['YrSold'] - train_df['YearBuilt']

#train_df['avgPrice'] = train_df['SalePrice'] / train_df['GrLivArea']

#train_df.drop(['Id', 'YrSold', 'YearBuilt'], axis=1, inplace=True)

#train_df['TotalBathBsmt'] = train_df['BsmtFullBath'] + train_df['BsmtHalfBath']



train_df = train_df.loc[train_df['GrLivArea']<3500]

train_df = train_df.loc[train_df['GrLivArea']>500]

train_df = train_df.loc[train_df['SalePrice']<400000]

train_df = train_df.loc[train_df['LotArea']<20000]

train_df['LotFrontage'] = train_df['LotFrontage'].fillna(0)

train_df.info()
mszoning_dummies_train = pd.get_dummies(train_df['MSZoning'])

#mszoning_dummies_train.columns = ['Child','Female','Male']

#mszoning_dummies_train.describe()

#mszoning_dummies_train.drop(['MSZoning'], axis=1, inplace=True)



train_df = train_df.join(mszoning_dummies_train)

train_df.drop(['MSZoning'], axis=1, inplace=True)

#zoning_train_df = train_df['MSZoning']

#train_df[['MSZoning', 'Id']].groupby(['MSZoning'],as_index=False).count()



#train_df['MSZoning'].loc[train_df['MSZoning'] == "C (all)"] = 'C'

#train_df[['MSZoning', 'Id']].groupby(['MSZoning'],as_index=False).count()



#fig, (axis1) = plt.subplots(1,1,figsize=(10,5))

#sns.countplot(x='YearBuilt', data=train_df, ax=axis1)

test_df = pd.read_csv('../input/test.csv')

test_df = test_df[['Id', 'LotArea', 'TotalBsmtSF', 'GrLivArea','GarageCars', 'YearBuilt', 'YrSold', 'PoolArea', 'TotRmsAbvGrd', 'FullBath', 'HalfBath', 'KitchenAbvGr', 'BedroomAbvGr', 'GarageArea','BedroomAbvGr','BsmtFullBath','BsmtHalfBath', 'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','LotFrontage','BsmtUnfSF','LowQualFinSF','BsmtFinSF1','BsmtFinSF2', 'MSZoning']]



#test_df['TotalBathAbvGr'] = test_df['FullBath'] + test_df['HalfBath']

test_df['NumOfYear'] = test_df['YrSold'] - test_df['YearBuilt']

test_df_Id = test_df['Id']

test_df.drop(['Id', 'YrSold', 'YearBuilt'], axis=1, inplace=True)



#LotArea

avg_lotArea_test = test_df['LotArea'].mean()

std_lotArea_test = test_df['LotArea'].std()

count_outliner_lotArea_test = test_df['LotArea'].loc[test_df['LotArea']>20000].count()



rand_1 = np.random.randint(avg_lotArea_test - std_lotArea_test, avg_lotArea_test + std_lotArea_test, size = count_outliner_lotArea_test)

test_df["LotArea"][test_df['LotArea']>20000] = rand_1



avg_liveArea_test = test_df['GrLivArea'].mean()

test_df['GrLivArea'].loc[test_df['GrLivArea'] > 3500] = avg_liveArea_test





mean_BsmtSF = test_df['TotalBsmtSF'].mean()

mean_GarageCars = test_df['GarageCars'].mean()

mean_GarageArea = test_df['GarageArea'].mean()

mean_BsmtFullBath = test_df['BsmtFullBath'].mean()

mean_BsmtHalfBath = test_df['BsmtHalfBath'].mean()

mean_BsmtUnfSF = test_df['BsmtUnfSF'].mean()

mean_BsmtFinSF1 = test_df['BsmtFinSF1'].mean()

mean_BsmtFinSF2 = test_df['BsmtFinSF2'].mean()



test_df['TotalBsmtSF'] = test_df['TotalBsmtSF'].fillna(mean_BsmtSF)

test_df['GarageCars'] = test_df['GarageCars'].fillna(mean_GarageCars)

test_df['GarageArea'] = test_df['GarageArea'].fillna(mean_BsmtSF)

test_df['BsmtFullBath'] = test_df['BsmtFullBath'].fillna(mean_GarageCars)

test_df['BsmtHalfBath'] = test_df['BsmtHalfBath'].fillna(mean_BsmtSF)

test_df['BsmtUnfSF'] = test_df['BsmtUnfSF'].fillna(mean_GarageCars)

test_df['BsmtFinSF1'] = test_df['BsmtFinSF1'].fillna(mean_BsmtSF)

test_df['BsmtFinSF2'] = test_df['BsmtFinSF2'].fillna(mean_BsmtSF)

test_df['MSZoning'] = test_df['MSZoning'].fillna('RL')

test_df['LotFrontage'] = test_df['LotFrontage'].fillna(0)

test_df.info()
mszoning_dummies_test = pd.get_dummies(test_df['MSZoning'])

#mszoning_dummies_train.columns = ['Child','Female','Male']

#mszoning_dummies_test.drop(['MSZoning'], axis=1, inplace=True)



test_df = test_df.join(mszoning_dummies_test)

test_df.drop(['MSZoning'], axis=1, inplace=True)



#test_df[['MSZoning', 'LotArea']].groupby(['MSZoning'],as_index=False).count()



#salePrice_df = train_df[['Id', 'SalePrice']].groupby('SalePrice').count()

#test_df = test_df.loc[test_df['LotArea']<20000]



#test_lotArea_median = test_df['LotArea'].median()

#test_df['LotArea'].loc[test_df['LotArea']>20000] = test_lotArea_median

#test_df['GrLivArea'].plot(kind='hist', figsize=(15,3),bins=100)
#train_df.count()

#train_df['LotArea'].loc[train_df['LotArea']>40000].count()



X_train = train_df.drop(['SalePrice'], axis=1)

Y_train = train_df['SalePrice']

X_test = test_df.copy()
#ridge_reg = linear_model.Ridge(alpha = 0.01)

#ridge_reg.fit(X_train, Y_train)

#Y_pred = ridge_reg.predict(X_test)

#ridge_reg.score(X_train, Y_train)
lasso_reg = linear_model.Lasso(alpha = 0.01)

lasso_reg.fit(X_train, Y_train)

Y_pred = lasso_reg.predict(X_test)

lasso_reg.score(X_train, Y_train)
elast_reg = linear_model.ElasticNet(alpha=0.01, l1_ratio=0.7)

elast_reg.fit(X_train, Y_train)

Y_pred = elast_reg.predict(X_test)

elast_reg.score(X_train, Y_train)
scores = cross_validation.cross_val_score(lasso_reg, X_train, Y_train, cv=5)

scores
##random_forest = RandomForestClassifier(n_estimators=200)



#random_forest.fit(X_train, Y_train)



#Y_pred = random_forest.predict(X_test)



#random_forest.score(X_train, Y_train)
coeff_df = DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df["Coefficient Estimate"] = pd.Series(elast_reg.coef_)



# preview

coeff_df


#fig, (axis1, axis2) = plt.subplots(2,1,figsize=(20,8))



#train_df['newRooms'] = train_df['TotRmsAbvGrd']-train_df['TotalBathAbvGr']+train_df['TotalBathBsmt']-train_df['BedroomAbvGr']

#sns.pointplot(x='GrLivArea', y='SalePrice', data=train_df[['SalePrice', 'GrLivArea']].loc[train_df['GrLivArea'] <= 4000], ax=axis1)

#train_df['SalePrice'].plot(kind='line', ax=axis1)

#train_df['LotArea'].plot(kind='line', ax=axis1)

#train_df['GrLivArea'].plot(kind='line', ax=axis2)

#train_df['KitchenAbvGr'].plot(kind='line', ax=axis1, xlim=(0, 10))
submission = pd.DataFrame({

        "Id": test_df_Id,

        "SalePrice": Y_pred

    })

submission.to_csv('HousePricePred_20170104.csv', index=False)