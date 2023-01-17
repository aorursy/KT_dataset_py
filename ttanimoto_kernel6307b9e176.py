# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train.describe
import matplotlib.pyplot as plt
import seaborn as sns
train.corr()
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
Ddata = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
Ddata_t= pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns',100)
train.isnull().sum()
train.drop(['FireplaceQu','PoolQC','Fence','MiscFeature','Alley', 'LotFrontage'], axis=1, inplace=True)

train_ID=train["Id"]
test_ID=test["Id"]

y_train=train["SalePrice"]

X_train["TotalSF"]=X_train["TotalBsmtSF"]+X_train["1stFlrSF"]+X_train["2ndFlrSF"]
X_test["TotalSF"]=X_test["TotalBsmtSF"]+X_test["1stFlrSF"]+X_test["2ndFlrSF"]
X_train
sns.distplot(y_train)
plt.show()
sns.distplot(np.log(y_train))