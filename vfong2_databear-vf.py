# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline



from pandas import DataFrame, Series

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import statsmodels.api as sm

from sklearn.cross_validation import train_test_split



import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
local_train, local_test = train_test_split(train,test_size=0.2,random_state=123)
train.shape
train_dum = pd.get_dummies(train,drop_first=True)

test_dum = pd.get_dummies(test,drop_first=True)



train_dum.head()

test_dum.head()
train_dum.shape

test.shape
corr_table = train_dum.corr()

corr_table.shape

corr_sp = corr_table['SalePrice']

#print("List the numerical features decendingly by their correlation with Sale Price:\n")

#for ele in sorted(corr_sp.items(), key = lambda x: -abs(x[1])):

 #   print("{0}: \t{1}".format(*ele))
#correlation table

saleprice_corr = corr_sp.to_frame(name="correlation coefficient")

saleprice_corr.sort(["correlation coefficient"], ascending = False, inplace=True)

saleprice_pos_corr = saleprice_corr[saleprice_corr["correlation coefficient"] > 0]

saleprice_neg_corr = saleprice_corr[saleprice_corr["correlation coefficient"] < 0].sort(["correlation coefficient"])

saleprice_corr

sns.boxplot(train['OverallCond'], train['SalePrice'])
plt.scatter(train['SalePrice'], train['GrLivArea'])

plt.xlabel('Sale Price ($)')

plt.ylabel('Above grade (ground) living area (sq.ft)')

plt.title('Scatterplot of Sale Price vs. Living Area square feet')
train.groupby('OverallQual')['SalePrice'].mean()
sns.boxplot(train['OverallQual'],train['SalePrice'] )

plt.ylabel('Sale Price ($)')

plt.xlabel('Overall Quality')
sns.boxplot(train['Neighborhood'],train['SalePrice'] )

plt.ylabel('Sale Price ($)')

plt.xlabel('Neighborhood')
a = train.groupby('Neighborhood')['SalePrice'].mean()

nsp = a.to_frame(name = 'Sale$')

nsp.sort(["Sale$"], ascending = False)
a = train.groupby('Heating')['SalePrice'].mean()

heating_sp = a.to_frame(name = "Sale Price")

heating_sp.sort(["Sale Price"], ascending = False)
#Pandas Histogram

train['SalePrice'].hist(bins=20)
sns.stripplot(x ='CentralAir', y="SalePrice", hue='Fireplaces', data=train, jitter= True)

sns.plt.title("Sale Prices with and without Air Conditioning")
import missingno as msno



#~/anaconda/bin/pip install missingno if using anaconda, otherwise just use pip install

msno.matrix(train_dum)