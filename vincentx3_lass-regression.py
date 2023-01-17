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
#pre-processing
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
data=pd.read_csv('../input/lasso_whole.csv')
data.describe()
x=data.drop(['year_county_drug','county_reports'],axis=1)
y=data.ix[:,'year_county_drug']
x.head()
toy_x=x[0:100]
toy_y=y[0:100]
toy_x.head()
#Lasso
clf_toy = linear_model.Lasso(alpha = 0.1,normalize=True)
clf_toy.fit(toy_x,toy_y)
X_train,X_test,y_train,y_test = train_test_split(x,y)
#Lasso with different alpha 
lasso_1 = linear_model.Lasso(alpha = 1,normalize=True)
lasso_1.fit(X_train,y_train)
lasso_05 = linear_model.Lasso(alpha = 0.5,normalize=True)
lasso_05.fit(X_train,y_train)
lasso_01 = linear_model.Lasso(alpha = 0.1,normalize=True)
lasso_01.fit(X_train,y_train)
lasso_001 = linear_model.Lasso(alpha = 0.01,normalize=True)
lasso_001.fit(X_train,y_train)
print('**********************************')
print("Lasso alpha=1")
print ("training set score:{:.2f}".format(lasso_1.score(X_train,y_train)))
print ("test set score:{:.2f}".format(lasso_1.score(X_test,y_test)))
print ("Number of features used:{}".format(np.sum(lasso_1.coef_!=0)))
print('**********************************')
print("Lasso alpha=0.5")
print ("training set score:{:.2f}".format(lasso_05.score(X_train,y_train)))
print ("test set score:{:.2f}".format(lasso_05.score(X_test,y_test)))
print ("Number of features used:{}".format(np.sum(lasso_05.coef_!=0)))
print('**********************************')
print("Lasso alpha=0.1")
print ("training set score:{:.2f}".format(lasso_01.score(X_train,y_train)))
print ("test set score:{:.2f}".format(lasso_01.score(X_test,y_test)))
print ("Number of features used:{}".format(np.sum(lasso_01.coef_!=0)))
print('**********************************')
print("Lasso alpha=0.01")
print ("training set score:{:.2f}".format(lasso_001.score(X_train,y_train)))
print ("test set score:{:.2f}".format(lasso_001.score(X_test,y_test)))
print ("Number of features used:{}".format(np.sum(lasso_001.coef_!=0)))
plt.figure(figsize = (7,7))
plt.plot(lasso_1.coef_,'s',label = "Lasso alpha=1")
plt.plot(lasso_01.coef_,'v',label = "Lasso alpha=0.5")
plt.plot(lasso_01.coef_,'^',label = "Lasso alpha=0.1")
plt.plot(lasso_001.coef_,'.',label = "Lasso alpha=0.01")
plt.xlabel('Coefficient index')
plt.ylabel('Coefficient magnitude')
plt.ylim(-25,25)
plt.legend(ncol=2,loc=(0,1.05))
plt.show()
#掩码提取特征
mask=lasso_1.coef_!=0
features_extract_1=x.loc[:,mask]
print("alpha=1,extract features:")
features_extract_1.head()
mask=lasso_05.coef_!=0
features_extract_05=x.loc[:,mask]
print("alpha=0.5,extract features:")
features_extract_05.head()
mask=lasso_01.coef_!=0
features_extract_01=x.loc[:,mask]
print("alpha=0.1,extract features:")
features_extract_01.head()
