# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import seaborn as sns
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')
df.shape
df.isna().sum()
df['SmokingStatus'].unique()
sns.countplot(df['SmokingStatus'],hue=df['Sex'])
df['Weeks']=np.where(df['Weeks']<0,-(df['Weeks']),df['Weeks'])
sns.distplot(df['FVC'])
sns.boxplot(df.Percent)
sns.distplot(df.Percent)
print(df.Percent.describe())
value=df['Percent'].mean()+df['Percent'].std()*3
value
df['Percent'].median()
df['Percent']=np.where(df['Percent']>=120,df['Percent'].median(),df['Percent'])
sns.distplot(df['Percent'])
sns.boxplot(df['Percent'])
dummies=pd.get_dummies(df['SmokingStatus'])
df=pd.concat([df,dummies],axis=1)
df
X=df.drop(['Patient','SmokingStatus','Sex'],axis=1)
y=df['FVC']
model=LinearRegression()
model.fit(X,y)
model.score(X,y)
df1=pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv')
X_test=df.drop(['Patient','SmokingStatus','Sex'],axis=1)
y_test=df['FVC']
y_pred=model.predict(X)
from sklearn.metrics import r2_score
r2_score(y,y_pred)
import statsmodels.api as sm
regressor_ols=sm.OLS(endog=y,exog=X).fit()
regressor_ols.summary()
model1=RandomForestRegressor()
model1.fit(X,y)
y_pred1=model1.predict(X_test)
model1.score(X,y)
