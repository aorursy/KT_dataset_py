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
df1=pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df1
df1.head
df1.tail
df1.info()
df1.describe()
df1.boxplot()
x=df1.iloc[:,:-1]

y=df1.iloc[:,-1]
x.shape
y.shape
from sklearn.model_selection import train_test_split

xT,xt,yT,yt=train_test_split(x,y,test_size=0.25,random_state=5)
xT.shape
xt.shape
yT.shape
yt.shape
from sklearn.linear_model import LinearRegression

lm=LinearRegression()

lm.fit(xT,yT)
p=lm.predict(xT)
p
yT
from sklearn.metrics import mean_squared_error,r2_score

mse=mean_squared_error(yT,p)

r_squared=r2_score(yT,p)
from math import sqrt

rmse=sqrt(mse)
print('Mean_Squared_Error:',mse)

print('Root_Mean_Squared_Error:',rmse)

print('r_square_value:',r_squared)
pf=lm.predict(xt)
pf
yt
from sklearn.metrics import mean_squared_error,r2_score

mse=mean_squared_error(yt,pf)

r_squared=r2_score(yt,pf)
print('Mean_Squared_Error1:',mse)

print('Root_Mean_Squared_Error1:',rmse)

print('r_square_value1:',r_squared)