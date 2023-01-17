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
x=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

x_test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')
x.head()
x.describe()
x.isnull().sum()
x.corr()
x1=x.iloc[:,3:5]

x_test2=x_test.iloc[:,3:5]
x_test.isnull().sum()
x1
from fancyimpute import IterativeImputer as MICE
x1_c = list(x1)
x1_c
y=MICE()
x2 = pd.DataFrame(y.fit_transform(x1))
x_test3=pd.DataFrame(y.transform(x_test2))
x2
x2.columns=x1_c

x_test3.columns=x1_c
x2.isnull().sum()
x_test3.isnull().sum()
x['Lat']=x2['Lat']

x_test['Lat']=x_test3['Lat']
x['Long']=x2['Long']

x_test['Long']=x_test3['Long']
x.isnull().sum()
x['Date']=pd.to_datetime(x['Date'])

x_test['Date']=pd.to_datetime(x_test['Date'])
x['Date'].describe()
from datetime import date
mindate=x['Date'].min()
mindate
z=x['Date']-mindate
for i in range(len(z)):

    z[i]=int(str(z[i]).split()[0])
x['Date']=z
x
c=x_test['Date']-mindate
for i in range(len(c)):

    c[i]=int(str(c[i]).split()[0])
c
x_test['Date']=c
x['Date']=x['Date'].astype(int)
x_test['Date']=x_test['Date'].astype(int)
x_test
y1_val=x.iloc[:,-2]
y1_val=y1_val.astype(int)
y2_val=x.iloc[:,-1]
y2_val=y2_val.astype(int)
x_val=x.drop(['ConfirmedCases','Fatalities'],axis=1)
x_val
z=x.corr()
import seaborn as sns;
sns.heatmap(z)
# for i in range(len(x)):

z=x_val['Province/State'].isna()

z1=x_test['Province/State'].isna()
for i in range(len(z1)):

       if z1[i]==True:

            x_test['Province/State'][i]=x_test['Country/Region'][i]
x_test.isna().sum()
for i in range(len(z)):

       if z[i]==True:

            x_val['Province/State'][i]=x_val['Country/Region'][i]
x_val.isnull().sum()
x.drop(['ConfirmedCases','Fatalities'],axis=1,inplace=True)
x
x_val.isnull().sum()
xx=pd.get_dummies(x_val)
x_predfinal=pd.get_dummies(x_test)
x_predfinal
xx
sns.heatmap(xx.corr())
# xx.to_csv('hihi.csv')
from sklearn.ensemble import BaggingRegressor
from sklearn import tree


from sklearn.model_selection import train_test_split

X_train1, X_test1, y_train1, y_test1 = train_test_split(xx, y1_val, test_size=0.2)

X_train2, X_test2, y_train2, y_test2 = train_test_split(xx, y2_val, test_size=0.2)
X_test1
model = BaggingRegressor(tree.DecisionTreeRegressor(random_state=1))
model.fit(X_train1, y_train1)
model.score(X_test1,y_test1)
# mio=pd.read_csv('')
anssss=model.predict(x_predfinal)
anssss
a1=np.around(anssss)
a1=a1.astype(int)
a1
model2 = BaggingRegressor(tree.DecisionTreeRegressor(random_state=1))
model2.fit(X_train2, y_train2)
model2.score(X_test2,y_test2)
anzzzz=model2.predict(x_predfinal)
a2=np.around(anzzzz)
a2=a2.astype(int)
predictions = pd.DataFrame({'ForecastId':x_test['ForecastId'],'ConfirmedCases':a1,'Fatalities':a2})

predictions.head()
# predictions.to_csv('submission.csv', header=True, index=False)
from sklearn.ensemble import AdaBoostRegressor
model3 = AdaBoostRegressor()

model4 = AdaBoostRegressor()
model3.fit(X_train1, y_train1)
model4.fit(X_train2, y_train2)
model3.score(X_test1,y_test1)
model4.score(X_test2,y_test2)
import xgboost as xgb
model5=xgb.XGBRegressor()
model6=xgb.XGBRegressor()
X_train1.rename(columns = {'Id':'ForecastId'}, inplace = True) 

X_test1.rename(columns = {'Id':'ForecastId'}, inplace = True) 
X_train2.rename(columns = {'Id':'ForecastId'}, inplace = True) 

X_test2.rename(columns = {'Id':'ForecastId'}, inplace = True) 
model5.fit(X_train1, y_train1)
model6.fit(X_train2, y_train2)
model5.score(X_test1,y_test1)
model6.score(X_test2,y_test2)
ank=model5.predict(x_predfinal)

annk=model6.predict(x_predfinal)



a4=np.around(ank)

a4=a4.astype(int)



a5=np.around(annk)

a5=a5.astype(int)



predictions2 = pd.DataFrame({'ForecastId':x_test['ForecastId'],'ConfirmedCases':a4,'Fatalities':a5})

predictions2.to_csv('submission.csv', header=True, index=False)

predictions2.head()
x_test
from sklearn.metrics import mean_squared_error
y_predmodel5=model5.predict(X_test1)
y_predmodel5
mean_squared_error(y_test1, y_predmodel5)