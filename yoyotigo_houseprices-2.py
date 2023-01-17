import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score


dataTesting = pd.read_csv('../input/train.csv')
testData = pd.read_csv('../input/test.csv')

dataTesting = dataTesting[~((dataTesting['GrLivArea'] > 4000) & (dataTesting['SalePrice'] < 300000))]

comboData = pd.concat((dataTesting.loc[:,'MSSubClass':'SaleCondition'],testData.loc[:,'MSSubClass':'SaleCondition']))

comboData.drop(['1stFlrSF', 'GarageArea', 'TotRmsAbvGrd'], axis=1, inplace=True)

dataTesting["SalePrice"] = np.log1p(dataTesting["SalePrice"])

numChar = comboData.dtypes[comboData.dtypes != "object"].index

skewChar = dataTesting[numChar].apply(lambda x: skew(x.dropna()))
skewChar = skewChar[skewChar > 0.65]
skewChar = skewChar.index

comboData[skewChar] = boxcox1p(comboData[skewChar], 0.15)
comboData = pd.get_dummies(comboData)
comboData = comboData.fillna(comboData.mean()) 

XdataTesting = comboData[:dataTesting.shape[0]]
XdataTest = comboData[dataTesting.shape[0]:]
y = dataTesting.SalePrice

lasso = Lasso(alpha=0.00003)
model = lasso
model.fit(XdataTesting, y)
preds = np.expm1(model.predict(XdataTest))

solution = pd.DataFrame({"id":testData.Id, "SalePrice":preds})
solution.to_csv("HousePrices.csv", index = False)# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
