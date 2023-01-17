# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
data = data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'MasVnrType'], axis=1)

test_data = test_data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'MasVnrType'], axis=1)
from sklearn import tree

from sklearn.linear_model import LinearRegression

# clf = tree.DecisionTreeClassifier()

# clf.fit(data.drop(['SalePrice'], axis=1), data['SalePrice'])

clf = LinearRegression().fit(data.drop(['SalePrice'], axis=1), data['SalePrice'])
predictions = clf.predict(test_data)

df_pred = pd.DataFrame(predictions, index=test_data.Id, columns=['SalePrice'])

df_pred
# import seaborn as sns

# %matplotlib inline

# sns.pairplot(data)

# print("sss")
# data.info()
data.head(5)
def replace_string_data_with_uniq_num(data_set, feature_name):

    count = 1

    for u in data_set[feature_name].unique():

        data_set[feature_name][data_set[feature_name] == u] = count

        count += 1
non_num_features = ['MSZoning', 'LandContour', 'Street','LotShape','Utilities','LotConfig','LandSlope','Neighborhood',

                    'Condition1','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','ExterQual',

                    'ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating',

                    'HeatingQC','CentralAir','HeatingQC','Electrical','Functional','GarageType','GarageFinish','GarageQual',

                    'GarageCond','PavedDrive','SaleType','SaleCondition', 'Condition2', 'KitchenQual']



for feature in non_num_features:

    replace_string_data_with_uniq_num(test_data, feature)

    replace_string_data_with_uniq_num(data, feature)



data.fillna(-1, inplace=True)

test_data.fillna(-1, inplace=True)