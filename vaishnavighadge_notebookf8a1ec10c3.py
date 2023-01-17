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
pd.pandas.set_option('display.max_columns',None)

df=pd.read_csv('/kaggle/input/weatherww2/Summary of Weather.csv')
df.head()
df.describe()
## 1 -step make the list of features which has missing values

missing=[features for features in df.columns if df[features].isnull().sum()>1]

## 2- step print the feature name and the percentage of missing values



for feature in missing:

    print(feature, np.round(df[feature].isnull().mean(), 3)*100,  ' % missing values')
df.isnull().sum()
cols=[feature for feature in df.columns  if(df[feature].isnull().sum()/df.shape[0]*100<70)]

new_df=df[cols]

new_df=new_df.drop(['STA'],axis=1)

print('actual columns after dropping null values are  %s'%new_df.shape[1])
new_df.isnull().sum()
new_df.dtypes
new_df.Date.unique()
new_df['Date']=pd.to_datetime(new_df['Date'])

new_df['Date'].head(300)
new_df['Snowfall'].unique()
new_df['SNF'].unique()
new_df['Precip']=pd.to_numeric(new_df['Precip'],errors='coerce')

new_df['Snowfall']=pd.to_numeric(new_df['Snowfall'],errors='coerce')

new_df['PRCP']=pd.to_numeric(new_df['PRCP'],errors='coerce')

new_df['SNF']=pd.to_numeric(new_df['SNF'],errors='coerce')

new_df.head()
new_df.columns
from sklearn.preprocessing import minmax_scale



new_df['Precip_scaled'] = minmax_scale(new_df['Precip'])

new_df['MaxTemp_scaled'] = minmax_scale(new_df['MaxTemp'])

new_df['MinTemp_scaled'] = minmax_scale(new_df['MinTemp'])

new_df['YR_scaled'] = minmax_scale(new_df['YR'])

new_df['MAX_scaled'] = minmax_scale(new_df['MAX'])

new_df['Snowfall_scaled'] = minmax_scale(new_df['Snowfall'])





new_df.columns
#plot the graphs

import matplotlib.pyplot as plt

import seaborn as sns
fig,ax=plt.subplots(4,2,figsize=(15,15))

sns.distplot(new_df['Precip'],ax=ax[0][0])

sns.distplot(new_df['Precip_scaled'],ax=ax[0][1])



sns.distplot(new_df['MaxTemp'],ax=ax[1][0])

sns.distplot(new_df['MaxTemp_scaled'],ax=ax[1][1])



sns.distplot(new_df['MinTemp'],ax=ax[2][0])

sns.distplot(new_df['MinTemp_scaled'],ax=ax[2][1])





sns.distplot(new_df['MAX'],ax=ax[3][0])

sns.distplot(new_df['MAX_scaled'],ax=ax[3][1])







from scipy.stats import boxcox



Precip_norm = boxcox(new_df['Precip_scaled'].loc[new_df['Precip_scaled'] > 0])

#MeanTemp_norm = boxcox(new_df['MeanTemp_scaled'].loc[new_df['MeanTemp_scaled'] > 0])



YR_norm = boxcox(new_df['YR_scaled'].loc[new_df['YR_scaled'] > 0])

Snowfall_norm = boxcox(new_df['Snowfall_scaled'].loc[new_df['Snowfall_scaled'] > 0])



MAX_norm = boxcox(new_df['MAX_scaled'].loc[new_df['MAX_scaled'] > 0])

#MIN_norm = boxcox(new_df['MIN_scaled'].loc[new_df['MIN_scaled'] > 0])
new_df.dtypes
import statsmodels

fig, ax = plt.subplots(4, 2, figsize=(15, 15))



sns.distplot(new_df['Precip_scaled'], ax=ax[0][0],kde=False)

sns.distplot(Precip_norm[0], ax=ax[0][1],kde=False)





sns.distplot(new_df['Snowfall_scaled'], ax=ax[1][0],kde=False)

sns.distplot(Snowfall_norm[0], ax=ax[1][1],kde=False)



sns.distplot(new_df['MAX_scaled'], ax=ax[2][0],kde=False)

#sns.distplot(MAX_norm[0], ax=ax[2][1],kde=False)
from scipy.stats import boxcox



Precip_norm = boxcox(new_df['Precip_scaled'].loc[new_df['Precip_scaled'] > 0])



YR_norm = boxcox(new_df['YR_scaled'].loc[new_df['YR_scaled'] > 0])

Snowfall_norm = boxcox(new_df['Snowfall_scaled'].loc[new_df['Snowfall_scaled'] > 0])

MAX_norm = boxcox(new_df['MAX_scaled'].loc[new_df['MAX_scaled'] > 0])

#hnadle NAN values

new_df.interpolate(method='linear',inplace=True)

new_df
new_df.isnull().sum()
new_df.head()
new_df.plot(x='MaxTemp_scaled',y='MinTemp_scaled',style='*')

plt.title('temperature')

plt.xlabel('min temp')

plt.ylabel('max temp')

plt.show()
#because sklearn expects a 2D array as input

X = new_df['MinTemp_scaled'].values.reshape(-1,1)

y = new_df['MaxTemp_scaled'].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
X_train.shape
y_train.shape
from sklearn.linear_model import LinearRegression
reg=LinearRegression()

reg
reg.fit(X_train,y_train)
#To retrieve the intercept:

print(reg.intercept_)



#For retrieving the slope:

print(reg.coef_) 

y_pred=reg.predict(X_test)

y_pred
from sklearn.metrics import confusion_matrix

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))