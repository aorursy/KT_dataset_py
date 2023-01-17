# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn 

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv(r'/kaggle/input/bike-sharing-demand/train.csv')

test=pd.read_csv(r'/kaggle/input/bike-sharing-demand/test.csv')

sub=pd.read_csv(r'/kaggle/input/bike-sharing-demand/sampleSubmission.csv')
sub.head()
test.head()
train.head()
dt = test["datetime"]
train['datetime'] = pd.to_datetime(train['datetime'])

train['Hour'] = train['datetime'].apply(lambda x:x.hour)

train['Month'] = train['datetime'].apply(lambda x:x.month)

train['Day of Week'] = train['datetime'].apply(lambda x:x.dayofweek)

train['year'] = [t.year for t in pd.DatetimeIndex(train.datetime)]

train['year'] = train['year'].map({2011:0, 2012:1})
test['datetime'] = pd.to_datetime(test['datetime'])

test['Hour'] = test['datetime'].apply(lambda x:x.hour)

test['Month'] = test['datetime'].apply(lambda x:x.month)

test['Day of Week'] = test['datetime'].apply(lambda x:x.dayofweek)

test['year'] = [t.year for t in pd.DatetimeIndex(test.datetime)]

test['year'] = test['year'].map({2011:0, 2012:1})
train.head()
test.head()
train=train.drop('datetime',axis=1)
train.head()
sns.countplot(x="year", data=train)

plt.show()
train["year"]=train['year'].replace(0,2011)

train["year"]=train['year'].replace(1,2012)
sns.countplot(x="year", data=train)

plt.show()
sns.barplot(x='Month',y='count',data=train)
sns.barplot(x='Hour',y='count',data=train)
sns.barplot(x='Day of Week',y='count',data=train)
plt.figure(figsize=(15,15))

sns.heatmap(train.corr(),annot=True)
sns.scatterplot(x="temp", y="atemp", data=train, hue="count")

plt.show()
sns.scatterplot(x="temp", y="count", data=train)

plt.show()
new_df=train.copy()

new_df.temp.describe()

new_df['temp_bin']=np.floor(new_df['temp'])//5

new_df['temp_bin'].unique()

# now we can visualize as follows

sns.factorplot(x="temp_bin",y="count",data=new_df,kind='bar')
train.head()
 # seperating season as per values. this is bcoz this will enhance features.

season=pd.get_dummies(train['season'],prefix='season')

train=pd.concat([train,season],axis=1)

train.head()
 # seperating season as per values. this is bcoz this will enhance features.

weather=pd.get_dummies(train['weather'],prefix='weather')

train=pd.concat([train,weather],axis=1)

train.head()
train.head()
train.drop(['casual','registered'],inplace=True,axis=1)

train.head()
train.drop(['season','weather'],inplace=True,axis=1)


from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor



#model selection

from sklearn.model_selection import train_test_split,cross_validate

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

from sklearn.metrics import accuracy_score
x_train,x_test,y_train,y_test=train_test_split(train.drop('count',axis=1),train['count'],test_size=0.25,random_state=42)
train.head()
train.shape
test.head()
from sklearn.ensemble import RandomForestRegressor 

  

 # create regressor object 

regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 

  

# fit the regressor with x and y data 

regressor.fit(x_train,y_train)  
y_pred=regressor.predict(x_test)
y_pred
r2=metrics.r2_score(y_test,y_pred)
print(r2)
params = {'n_estimators': 500,

          'max_depth': 4,

          'min_samples_split': 5,

          'learning_rate': 0.01,

          'loss': 'ls'}
from sklearn.ensemble import GradientBoostingRegressor

reg = GradientBoostingRegressor(**params)

reg.fit(x_train, y_train)



mse =  reg.predict(x_test)

print(mse)
test.head()
test=test.drop('datetime',axis=1)
test.head()
season=pd.get_dummies(test['season'],prefix='season')

test=pd.concat([test,season],axis=1)



 # seperating season as per values. this is bcoz this will enhance features.

weather=pd.get_dummies(test['weather'],prefix='weather')

test=pd.concat([test,weather],axis=1)

train.head()
test.drop(['season','weather'],inplace=True,axis=1)

test.head()
test.shape
X_test1=test.iloc[:,:].values

X_test1.shape
pred=regressor.predict(X_test1)
pred=pred.reshape(-1,1)
pred
pred = pd.DataFrame(pred, columns=['count'])
df = pd.concat([dt, pred],axis=1)
df.head()
df['count'] = df['count'].astype('int')
df.to_csv('submission1.csv' , index=False)