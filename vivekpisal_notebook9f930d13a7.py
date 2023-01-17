# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,r2_score
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import datetime as dt
from sklearn.model_selection import train_test_split
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/ipl-dataset/ipl.csv')
df.shape
df.head()
plt.figure(figsize=(55,8))
df['batsman'].value_counts().plot(kind='bar')
batsman_rating=df['batsman'].value_counts().to_dict()
df['batsman_rating']=df['batsman'].map(batsman_rating)
plt.figure(figsize=(55,8))
df['bowler'].value_counts().plot(kind='bar')
bowler_rating=df['bowler'].value_counts().to_dict()
df['bowler_rating']=df['bowler'].map(bowler_rating)
df['date']=df['date'].apply(lambda x:dt.datetime.strptime(x,'%Y-%m-%d'))
df.head()
df['bat_team'].unique()
consitent_teams=['Kolkata Knight Riders','Chennai Super Kings','Rajasthan Royals','Mumbai Indians','Kings XI Punjab',
       'Royal Challengers Bangalore', 'Delhi Daredevils','Sunrisers Hyderabad']
df=df[df['bat_team'].isin(consitent_teams) & df['bowl_team'].isin(consitent_teams)]
df=pd.get_dummies(data=df,columns=['bat_team','bowl_team'])
df.head()
df=df[df['overs']>=5.0]
df.head()
stadium=df['venue'].value_counts().to_dict()
stadium
df['venue']=df['venue'].map(stadium)
df['date'].dt.year.unique()
df['date'].dt.year.value_counts()
X=df.drop(['total','mid','date','striker','non-striker','batsman','bowler','batsman_rating','bowler_rating'],axis=1)
y=df['total']
X.head()
plt.figure(figsize=(10,5))
sns.heatmap(X.corr())
plt.show()
X_train=X[df['date'].dt.year<=2016]
X_test=X[df['date'].dt.year>=2017]
y_train=y[df['date'].dt.year<=2016]
y_test=y[df['date'].dt.year>=2017]
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
result=sm.OLS(y,X).fit()
result.summary()
np.mean(y_pred-y_test)**2
model.score(X_test,y_test)
