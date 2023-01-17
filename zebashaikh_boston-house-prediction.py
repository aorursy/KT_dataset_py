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
from sklearn.datasets import load_boston
dataset= load_boston()
dataset.DESCR
dataset
df=pd.DataFrame(data=dataset.data,columns=dataset.feature_names)



df['MEDV']=dataset.target

df.head()
df.isna().any()
import seaborn as sb
sb.pairplot(df)
from sklearn import model_selection

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
X=df.loc[:,'CRIM':'LSTAT']

X.head()

y=df['MEDV']
xtrain,xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.3,random_state=42)
lm=LinearRegression()

lm.fit(xtrain,ytrain)

print(lm.intercept_)

print(lm.coef_)
preds=lm.predict(xtest)
from math import sqrt
print(sqrt(mean_squared_error(ytest,preds)))