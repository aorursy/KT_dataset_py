# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet



from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler



from sklearn.model_selection import train_test_split

import os

filename = os.listdir("../input")



df_train = pd.read_csv('../input/train.csv')

data_train = pd.DataFrame(df_train)

df_test = pd.read_csv('../input/test.csv')

data_test = pd.DataFrame(df_test)

# Any results you write to the current directory are saved as output.
print(df_train.columns)
df_train.shape
df_train.info()
fig,ax = plt.subplots(figsize=(20, 15))

sns.heatmap(df_train.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")

plt.show()
corr = df_train.corr()

most_corr_features = corr.index[abs(corr["SalePrice"])>0.5]

plt.figure(figsize=(10,10))

sns.heatmap(df_train[most_corr_features].corr(), annot=True, cmap="RdYlGn")

sns.pairplot(df_train[most_corr_features], height = 2.5)

plt.show()
df_train[df_train.GrLivArea >= 4000]
df_train = df_train[df_train.GrLivArea < 4000]
df_train[df_train.GrLivArea >= 4000]
sns.stripplot(x = df_train.Neighborhood.values, y = df_train.SalePrice.values,

             order = np.sort(df_train.Neighborhood.unique()),

             jitter=0.1, alpha=0.25)

plt.xticks(rotation=90)
cat_features = df_train.select_dtypes(include=['object']).columns

#print(cat_features)

num_features = df_train.select_dtypes(exclude=['object']).columns

num_features = num_features.drop('SalePrice')

#print(num_features)
plt.scatter(df_train["SaleType"], df_train["SalePrice"], alpha=0.1)


i = 0

while i < len(cat_features) - 5:

    g = sns.pairplot(df_train, x_vars=[cat_features[i], cat_features[i+1], cat_features[i+2], cat_features[i+3], cat_features[i+4]], y_vars=["SalePrice"], height = 3.5, aspect=1.0)

    for ax in g.axes[-1, :]:

        #ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        for lab in ax.get_xticklabels():

            lab.set_rotation(90)

    i += 5



final_features = most_corr_features

final_features = ['OverallQual', 'YearBuilt', 'YearRemodAdd',

       'GrLivArea', 'FullBath', 'GarageCars', 'GarageArea']

print(final_features)



x_train, x_test, y_train, y_test = train_test_split(df_train[final_features], df_train["SalePrice"], train_size=0.8, test_size = 0.2, random_state=3)

lm = LinearRegression()

lm.fit(x_train, y_train)

print(lm.score(x_train, y_train))

print(lm.score(x_test, y_test))



df_test = pd.read_csv('../input/test.csv')



null_vals = df_test[df_test['GarageArea'].isnull()]

#print(null_vals)

df_test.shape

df_test.set_value(1116,'GarageCars', 2)

df_test.set_value(1116,'GarageArea', 472)

df_test[df_test.Id == 2577]



'''df_test[final_features].info()

predictions = lm.predict(df_test[final_features])

print(predictions)

my_sub = pd.DataFrame({'Id':df_test.Id, 'SalePrice':predictions})

my_sub.to_csv('submission.csv', index = False)'''

print(df_train.info())


df_train["HasCircuitBreaker"] = df_train["Electrical"].map({'SBrkr':1, 'FuseA':-0.5, 'FuseF':-0.75,'FuseP':-0.5,'Mix':-0.5})

df_train.set_value(1379,'HasCircuitBreaker', df_train.HasCircuitBreaker.mean())

null_vals = df_train[df_train['Electrical'].isnull()]

print(null_vals)

#{'SBrkr':3.5, 'FuseA':2.5, 'FuseF':1.8,'FuseP':1.8,'Mix':0.5}
df_train["ExteriorCondGood"] = df_train["ExterCond"].map({'TA':1, 'Gd':0.75, 'Fa':0.5,'Ex':1,'Po':0.1})

#final_features.remove("HasCircuitBreaker")
#final_features.append('HasCircuitBreaker')

print(final_features)

x_train, x_test, y_train, y_test = train_test_split(df_train[final_features], df_train["SalePrice"], train_size=0.8, test_size = 0.2, random_state=3)

lm = LinearRegression()

lm.fit(x_train, y_train)

print(lm.score(x_train, y_train))

print(lm.score(x_test, y_test))
df_test['TotalBsmtSF'] = df_test['TotalBsmtSF'].fillna(df_test.TotalBsmtSF.mean())

df_train['MasVnrArea']= df_train['MasVnrArea'].fillna(df_train.MasVnrArea.mean())

df_test['MasVnrArea']= df_test['MasVnrArea'].fillna(df_test.MasVnrArea.mean())

df_test['GarageCars']= df_test['GarageCars'].fillna(df_test.GarageCars.mean())

df_test['BsmtFinSF1']= df_test['BsmtFinSF1'].fillna(df_test.BsmtFinSF1.mean())
#most_corr_features = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'TotalBsmtSF', 'GrLivArea', 'FullBath', 'Fireplaces', 'GarageCars']

most_corr_features = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', 'GrLivArea', 'FullBath', 'Fireplaces', 'GarageCars']

#x_train2, x_test2, y_train2, y_test2 = train_test_split(df_train[most_corr_features], df_train["SalePrice"], train_size=0.8, test_size=0.2, random_state=3)
df_train.select_dtypes(exclude='object').isnull().sum()
df_train['SalePrice'] = np.log1p(df_train['SalePrice'])*100
df_train['SalePrice']
steps = [

    

    ('poly', PolynomialFeatures(degree=3)),

    ('model', Lasso(alpha=400000, fit_intercept=True))

]

pipeline = Pipeline(steps)

x = df_train[most_corr_features]

y = df_train["SalePrice"]





x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=3)





pipeline.fit(x_train2, y_train2)

#pipeline.fit(df_train[most_corr_features], df_train['SalePrice'])

print(pipeline.score(x_train2, y_train2))

print(pipeline.score(x_test2, y_test2))



plt.scatter(pipeline.predict(x_test2), y_test2, alpha=0.3)

plt.plot([0, 800000], [0, 800000], '--r')
'''houses = pd.concat([train,test], sort=False)

houses = pd.get_dummies(houses)

df_train= houses[len_train]'''
predictions = pipeline.predict(df_test[most_corr_features])

final_predictions = np.expm1(predictions/100)

print(predictions)

my_sub = pd.DataFrame({'Id':df_test.Id, 'SalePrice':final_predictions})

my_sub.to_csv('submission.csv', index = False)