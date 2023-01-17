import numpy as np 

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns
dataset = pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv')
dataset.describe()
dataset.isnull().sum()
dataset.head()
dataset = dataset.drop('State', axis=1)
sns.distplot(dataset['AvgTemperature'])
dataset['AvgTemperature']= (dataset.AvgTemperature-32)*5/9
sns.distplot(dataset['AvgTemperature'])
sns.distplot(dataset['Month'])
sns.distplot(dataset['Day'])
sns.distplot(dataset['Year'])
sns.set(style="whitegrid")

sns.boxplot(x=dataset['Year'])
dataset=dataset[dataset['Year']>1990]

sns.boxplot(x=dataset['Year'])
sns.boxplot(x=dataset['Region'], y=dataset['AvgTemperature'], data=dataset)
sns.set(style="whitegrid")



chart = sns.barplot(x=dataset['Region'], y=dataset['AvgTemperature'],data=dataset)

chart.set_xticklabels(labels=chart.get_xticklabels(),rotation=90, horizontalalignment='left')

ax = sns.lineplot(x=dataset['Year'], y=dataset['AvgTemperature'], data=dataset)
dataset2 = dataset[dataset.Year<2020]
ax = sns.lineplot(x=dataset2['Year'], y=dataset2['AvgTemperature'], data=dataset2)
fig, ax = plt.subplots(figsize=(10, 8))

region = dataset['Region']

chart = sns.lineplot(x=dataset2['Year'], y=dataset2["AvgTemperature"], hue=region, legend="full",

                  data=dataset2)

chart.set_xticklabels(labels=chart.get_xticklabels(),rotation=90, horizontalalignment='left')

plt.show()
dataset['Weather_type']=dataset['AvgTemperature']
temp=[dataset]

for data in temp:

    data.loc[dataset['Weather_type']<=12,'Weather_type']=0,

    data.loc[(dataset['Weather_type']>12) & (dataset['Weather_type']<=18),'Weather_type']=1,

    data.loc[(dataset['Weather_type']>18) & (dataset['Weather_type']<=26),'Weather_type']=2,

    data.loc[(dataset['Weather_type']>26) & (dataset['Weather_type']<=35),'Weather_type']=3,

    data.loc[(dataset['Weather_type']>35),'Weather_type']=4
sns.distplot(dataset['Weather_type'])


sns.lineplot(x=dataset['Month'], y=dataset['AvgTemperature'], data=dataset)
fig, ax = plt.subplots(figsize=(10, 8))

sns.lineplot(x=dataset['Month'], y=dataset['AvgTemperature'], hue=dataset['Region'],legend="full", data=dataset)

plt.show()
dataset.drop('Weather_type',axis=1,inplace=True)
train = dataset[dataset.Year<2019]

test = dataset[dataset.Year>=2019]
#function to create new attribute

def attribute(df,col1,col2):

    index_col1=df.columns.get_loc(col1)

    index_col2=df.columns.get_loc(col2)

    for row in range(len(df)):

        f=df.iat[row,index_col1]

        df.iat[row,index_col2]=f



#calculating monthly average temperature for each city

g=train.groupby(['City','Month'])

df=g.mean()

df['AvgTemperature_City_Monthly']=0.0

attribute(df,'AvgTemperature','AvgTemperature_City_Monthly')

drop=['Year','Day', 'AvgTemperature']

df.drop(drop,axis=1,inplace=True)

on=['City','Month']

train=pd.merge(train,df,on=on,how='inner')

test=pd.merge(test,df,on=on,how='inner')





#calculating monthly average temperature for each country

g=train.groupby(['Country','Month'])

df=g.mean()

df['AvgTemperature_Country_Monthly']=0.0

attribute(df,'AvgTemperature','AvgTemperature_Country_Monthly')

drop=['Year','Day', 'AvgTemperature'

     ,'AvgTemperature_City_Monthly']

df.drop(drop,axis=1,inplace=True)

on=['Country','Month']

train=pd.merge(train,df,on=on,how='inner')

test=pd.merge(test,df,on=on,how='inner')
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

def encode(df):

    df['City'] = le.fit_transform(df['City'])

    df['Region'] = le.fit_transform(df['Region'])

    df['Country'] = le.fit_transform(df['Country'])

    

encode(train)

encode(test)
corr = train.corr()

fig, ax = plt.subplots(figsize=(10, 8))

colormap = sns.diverging_palette(220, 10, as_cmap=True)

dropSelf = np.zeros_like(corr)

dropSelf[np.triu_indices_from(dropSelf)] = True

colormap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=dropSelf)

plt.title('Temperatures - Features Correlations')

plt.show()
y_train=train['AvgTemperature']

X_train=train.drop('AvgTemperature',axis=1)



y_test=test['AvgTemperature']

X_test=test.drop('AvgTemperature',axis=1)
from sklearn.linear_model import Lasso

regr = Lasso(alpha=0.01)

regr.fit(X_train, y_train)

y_pred=regr.predict(X_test)
from sklearn.metrics import mean_squared_error

mse=np.sqrt(mean_squared_error(y_test,y_pred))



print('Root Mean Squared Error - {}'.format(mse))