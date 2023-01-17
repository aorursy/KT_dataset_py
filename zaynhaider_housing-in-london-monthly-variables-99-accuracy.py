
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# importing libraries
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.metrics import confusion_matrix
# reading dataset
ds = pd.read_csv('/kaggle/input/housing-in-london/housing_in_london_monthly_variables.csv')
ds.head()
ds.info()
# checking null values
ds.isnull().sum()
# filling null values with their corresponding mean of columns
mean_of_hs = ds['houses_sold'].mean()        
mean_of_noc = ds['no_of_crimes'].mean()

ds = ds.fillna({'houses_sold' : mean_of_hs})
ds = ds.fillna({'no_of_crimes' : mean_of_noc})
ds.isnull().sum()
from sklearn.preprocessing import LabelEncoder

area = LabelEncoder()
code = LabelEncoder()

ds['area_n'] = area.fit_transform(ds['area'])
ds['code_n'] = code.fit_transform(ds['code'])
ds.drop(['area','code','date'], axis = 1, inplace=True)
ds
ds['borough_flag'].value_counts()
borough_flag_1 = ds[ds['borough_flag']==1]
borough_flag_0 = ds[ds['borough_flag']==0]

borough_flag_1.shape , borough_flag_0.shape
borough_flag_1 = borough_flag_1.sample(n=borough_flag_0.shape[0])
borough_flag_1.shape
df = borough_flag_0.append(borough_flag_1, ignore_index=True)
df.shape
df['borough_flag'].value_counts()
x = df.drop('borough_flag', axis=1)
y = df['borough_flag']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25)
x_train.shape , x_test.shape
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt.score(x_test,y_test)
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
rf.score(x_test,y_test)
xgb = XGBClassifier()
xgb.fit(x_train,y_train)
xgb.score(x_test,y_test)
y_pred = rf.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n',cm)
plt.figure(figsize=(7,5))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('truth')
x = df.drop('houses_sold',axis=1)
y = df['houses_sold']

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25)
x_train.shape , x_test.shape
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
xgbr = XGBRegressor()
xgbr.fit(x_train,y_train)
xgbr.score(x_test,y_test)
dtr = DecisionTreeRegressor()
dtr.fit(x_train,y_train)
dtr.score(x_test,y_test)
knnr = KNeighborsRegressor()
knnr.fit(x_train,y_train)
knnr.score(x_test,y_test)
rfr = RandomForestRegressor()
rfr.fit(x_train,y_train)
rfr.score(x_test,y_test)
y_pred2 = knnr.predict(x_test)
y_pred2