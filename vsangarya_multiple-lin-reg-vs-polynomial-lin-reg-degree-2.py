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
ddf=pd.read_csv(r"/kaggle/input/diamonds/diamonds.csv")
ddf.head()
print(ddf['cut'].unique(),"  ", ddf['color'].unique(), "  ", ddf['clarity'].unique())
ddf.fillna(0,inplace = True)
cols = list(ddf.columns.values)
cols
ddf = ddf[['Unnamed: 0','carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z', 'price']]
ddf.drop(['Unnamed: 0'],axis=1,inplace=True)
ddf.head()
ddf.dtypes
from sklearn import preprocessing

X=ddf.values

cut=preprocessing.LabelEncoder()
cut.fit(['Ideal','Premium', 'Good', 'Very Good', 'Fair'])
X[:,1]=cut.transform(X[:,1])

color=preprocessing.LabelEncoder()
color.fit(['E', 'I', 'J', 'H', 'F', 'G', 'D'])
X[:,2]=color.transform(X[:,2])

clarity=preprocessing.LabelEncoder()
clarity.fit(['SI2', 'SI1', 'VS1', 'VS2', 'VVS2', 'VVS1', 'I1', 'IF'])
X[:,3]=clarity.transform(X[:,3])
X[0:1]
X.shape
x=X[:,0:9]
x=np.array(x, dtype=[('O', np.float)]).astype(np.float)

y=X[:,9]
y=np.array(y, dtype=[('O', np.float)]).astype(np.float)

from sklearn.model_selection import train_test_split
xlrtrain,xlrtest,ylrtrain,ylrtest=train_test_split(x,y,test_size=0.15,random_state=3)
print(xlrtrain.shape,ylrtrain.shape,xlrtest.shape,ylrtest.shape)
#Multiple Linear Regression

from sklearn import linear_model
MLR=linear_model.LinearRegression()
MLR.fit(xlrtrain,ylrtrain)
print("Thetas :",MLR.coef_)
print("intercept :",MLR.intercept_)
yhat=MLR.predict(xlrtest)
from sklearn.metrics import r2_score
print("R2 score %.5f" %r2_score(yhat,ylrtest))

#Polynomial Regression with degree 2
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
xp=poly.fit_transform(xlrtrain)
xtestp=poly.fit_transform(xlrtest)
MLRp=linear_model.LinearRegression()
MLRp.fit(xp,ylrtrain)
print("Thetas :",MLRp.coef_)
print("intercept :",MLRp.intercept_)
yhatp=MLRp.predict(xtestp)
print("R2 score %.5f" %r2_score(yhatp,ylrtest))

print("Mutiple Linear Regression R2 score %.5f" %r2_score(yhat,ylrtest))
print("Polynomial Regression R2 score %.5f" %r2_score(yhatp,ylrtest))

