# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
X= pd.read_csv('../input/train.csv')
X['SalePrice'].describe()
SalePrice = X['SalePrice']
sns.distplot(X['SalePrice']);
def plt_cont(var):

    var = 'GrLivArea'

    data = pd.concat([X['SalePrice'], X[var]], axis=1)

    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

    

    
plt_cont('GrLivSpace')
X.plot.scatter(x='GrLivArea',y='SalePrice');
X.plot.scatter(x='LotArea',y='SalePrice');
var = 'OverallQual'

data = pd.concat([X[var],X['SalePrice']],axis=1)

f,ax = plt.subplots(figsize = (8,6))

fig = sns.boxplot(x=var,y='SalePrice',data=X)

corr_mat = X.corr() 
fig,ax = plt.subplots(figsize=(10,8))

sns.heatmap(corr_mat)
k =10

cols = corr_mat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(X[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
sns.set()

cols = list(cols)

sns.pairplot(X[cols])

plt.show()



total = X.isnull().sum().sort_values(ascending=False)

percent = (X.isnull().sum()/X.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis =1,keys=['total','percent'])

missing_data.head(20)
sale_Price_Scaled=StandardScaler().fit_transform(X['SalePrice'][:,np.newaxis])
low_Price= sale_Price_Scaled[sale_Price_Scaled[:,0].argsort(),0][:10]

high_Price= sale_Price_Scaled[sale_Price_Scaled[:,0].argsort(),0][-10:]
print(low_Price)

print(high_Price)
var = 'GrLivArea'

data = pd.concat([X[var],X['SalePrice']],axis=1)

data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))