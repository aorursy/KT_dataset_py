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
import seaborn as sns
import pandas as pd

h_test = pd.read_csv("../input/housetest.csv")

h_train = pd.read_csv("../input/housetrain.csv")
h_train
h_test.head()
h_train.head()
h_train.isnull().sum()
h_train.describe()
h_train.columns
h_train.info()
X=h_train.drop(['Id','Alley','PoolQC','Fence','MiscFeature','SalePrice'],axis=1)
X
Y=h_train['SalePrice']
X_train.shape
X.isnull().sum()
X['LotFrontage'].mean()
X['LotFrontage'].fillna(X['LotFrontage'].mean(),inplace=True)
X['MasVnrArea'].fillna(X['MasVnrArea'].mean(),inplace=True)
X['GarageYrBlt'].fillna(X['GarageYrBlt'].mean(),inplace=True)
X.info()
X['MasVnrType'].value_counts()
X['MasVnrType'].fillna("None",inplace=True)
X['BsmtQual'].fillna(X['BsmtQual'].mode()[0],inplace=True)

X['BsmtCond'].fillna(X['BsmtCond'].mode()[0],inplace=True)

X['BsmtExposure'].fillna(X['BsmtExposure'].mode()[0],inplace=True)

X['BsmtFinType1'].fillna(X['BsmtFinType1'].mode()[0],inplace=True)

X['BsmtFinType2'].fillna(X['BsmtFinType2'].mode()[0],inplace=True)

X['Electrical'].fillna(X['Electrical'].mode()[0],inplace=True)

X['FireplaceQu'].fillna(X['FireplaceQu'].mode()[0],inplace=True)

X['GarageType'].fillna(X['GarageType'].mode()[0],inplace=True)

X['GarageFinish'].fillna(X['GarageFinish'].mode()[0],inplace=True)

X['GarageQual'].fillna(X['GarageQual'].mode()[0],inplace=True)

X['GarageCond'].fillna(X['GarageCond'].mode()[0],inplace=True)
X.info()
h_test.info()
h_test.drop(['Id','Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
h_test
def count(val):

    r=h_test[val].value_counts()

    print(r)

def fill(value):

    h_test[value].fillna(h_test[value].mode()[0],inplace=True)
col=h_test.columns.tolist()

col
for i in col:

    count(i)

    fill(i)
h_test.info()
h_test=pd.get_dummies(h_test)
h_test.shape
missing_cols = set( X_train.columns ) - set(h_test.columns )

missing_cols
for c in missing_cols:

    h_test[c] = 0
h_test.shape
sns.heatmap(X.corr())
X=pd.get_dummies(X)
X.head()
from sklearn.model_selection import train_test_split



X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.30)
import xgboost as xgb

xg_reg = xgb.XGBRegressor()

xg_reg.fit(X_train,Y_train)
xg_reg.score(X_test,Y_test)
xg_reg.predict(h_test)