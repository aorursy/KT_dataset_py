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
import pandas as pd

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test.head(1)
train.info()
train.head(3)
train=train.drop(['Id'], axis=1)
train.columns
train['1stFlrSF'].unique
test.columns
X_train=train.iloc[:,1:37].values
X_test=test.iloc[:,1:37].values
X_test
Y_train=train.iloc[:,37].values
Y_train
from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model = ExtraTreesClassifier()

model.fit(X_train,Y_train)

print(model.feature_importances_) 

feat_importances = pd.Series(model.feature_importances_, index=train.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()
from sklearn.linear_model import LinearRegression
model=LinearRegression()
from sklearn import preprocessing



def convert(train):

    number = preprocessing.LabelEncoder()

    train['LandSlope'] = number.fit_transform(train['LandSlope'])

    

    return train
model.fit(X_train, Y_train)
result= model.score(X_train, Y_train)

result
test.fillna(test.mean(), inplace=True)
predict_lin= model.predict(X_test)
predict_lin
submission_rfc = pd.DataFrame({

    'Id':test.Id.values

})

submission_rfc['SalePrice'] = predict_lin

submission_rfc.to_csv('submission_lin.csv', index=False)