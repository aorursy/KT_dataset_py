#data analysis libraries 
import numpy as np
import pandas as pd
import datetime

#visualization libraries
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
%matplotlib inline

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('../input/avocado.csv') #read to data
data = data.drop(['Unnamed: 0'], axis = 1) #drop the useless column
names = ["date", "avprice", "totalvol", "small","large","xlarge","totalbags","smallbags","largebags","xlargebags","type","year","region"] #get new column names
data = data.rename(columns=dict(zip(data.columns, names))) #rename columns
data.head()
data.info()
dates = [datetime.datetime.strptime(ts, "%Y-%m-%d") for ts in data['date']]
dates.sort()
sorteddates = [datetime.datetime.strftime(ts, "%Y-%m-%d") for ts in dates]
data['date'] = pd.DataFrame({'date':sorteddates})
data['Year'], data['Month'],  data['Day'] = data['date'].str.split('-').str
data.head(10)
plt.figure(figsize=(12,5))
plt.title("Price Distirbution Graph")
ax = sns.distplot(data["avprice"], color = 'y')
import seaborn as sns
fig, ax = plt.subplots()
fig.set_size_inches(10,5)
sns.violinplot(data.dropna(subset = ['avprice']).avprice)
dategroup=data.groupby('date').mean()
plt.figure(figsize=(12,5))
dategroup['avprice'].plot(x=data.date)
plt.title('Average Price')
dategroup=data.groupby('Month').mean()
fig, ax = plt.subplots(figsize=(12,5))
ax.xaxis.set(ticks=range(0,13)) # Manually set x-ticks
dategroup['avprice'].plot(x=data.Month)
plt.title('Average Price by Month')
dategroup=data.groupby('Day').mean()
fig, ax = plt.subplots(figsize=(12,5))
ax.xaxis.set(ticks=range(0,31)) # Manually set x-ticks
dategroup['avprice'].plot(x=data.Day)
plt.title('Average Price by Day')
plt.figure(figsize=(20,20))
sns.set_style('whitegrid')
sns.pointplot(x='avprice',y='region',data=data, hue='year',join=False)
plt.xticks(np.linspace(1,2,5))
plt.xlabel('Region',{'fontsize' : 'large'})
plt.ylabel('Average Price',{'fontsize':'large'})
plt.title("Yearly Average Price in Each Region",{'fontsize':20})
plt.figure(figsize=(12,20))
sns.set_style('whitegrid')
sns.pointplot(x='avprice',y='region',data=data, hue='type',join=False)
plt.xticks(np.linspace(1,2,5))
plt.xlabel('Region',{'fontsize' : 'large'})
plt.ylabel('Average Price',{'fontsize':'large'})
plt.title("Type Average Price in Each Region",{'fontsize':20})
print(data['type'].value_counts())
plt.figure(figsize=(12,5))
sns.countplot(data['type'])
plt.show()
%matplotlib inline
import pandas as pd
from fbprophet import Prophet

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
df = data.loc[:, ["date","avprice"]]
df['date'] = pd.DatetimeIndex(df['date'])
df.dtypes
df = df.rename(columns={'date': 'ds',
                        'avprice': 'y'})
ax = df.set_index('ds').plot(figsize=(20, 12))
ax.set_ylabel('Monthly Average Price of Avocado')
ax.set_xlabel('Date')

plt.show()
my_model = Prophet()
my_model.fit(df)

future_dates = my_model.make_future_dataframe(periods=900)
forecast =my_model.predict(future_dates)
fig2 = my_model.plot_components(forecast)
forecastnew = forecast['ds']
forecastnew2 = forecast['yhat']

forecastnew = pd.concat([forecastnew,forecastnew2], axis=1)

mask = (forecastnew['ds'] > "2018-03-24") & (forecastnew['ds'] <= "2020-09-10")
forecastedvalues = forecastnew.loc[mask]

mask = (forecastnew['ds'] > "2015-01-04") & (forecastnew['ds'] <= "2018-03-25")
forecastnew = forecastnew.loc[mask]
fig, ax1 = plt.subplots(figsize=(16, 8))
ax1.plot(forecastnew.set_index('ds'), color='b')
ax1.plot(forecastedvalues.set_index('ds'), color='r')
ax1.set_ylabel('Average Prices')
ax1.set_xlabel('Date')
print("Red = Predicted Values, Blue = Base Values")