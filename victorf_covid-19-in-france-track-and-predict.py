import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

raw_data = pd.read_csv( "../input/coronavirusdataset-france/chiffres-cles.csv", parse_dates=['date']) # raw dataframe

raw_data.rename(columns={'cas_confirmes':'cases', 'deces':'deaths'},inplace=True) #important variable namess in English
raw_data.head()
latest_date = max(raw_data['date'])
print(latest_date)
national_latest = raw_data[raw_data['date'] == latest_date]


df_national = raw_data[raw_data.maille_nom =='France']

df_national.reset_index(inplace = True, drop=True)
df_national = df_national[['date','cases','deaths']]
df_national = df_national.groupby(['date']).mean().reset_index() # get cases each day
print(df_national.dtypes)
df_national.tail()


df_national = df_national[df_national['date'] > '2020-03-01']
df_national.date = pd.to_datetime(df_national.date)
df_national.reset_index(inplace = True, drop=True)
#df_google.set_index('datetime', inplace=True)
y = df_national['cases'].values # transform the column to differentiate into a numpy array

deriv_y = np.gradient(y) # now we can get the derivative as a new numpy array

output = np.transpose(deriv_y)
#now add the numpy array to our dataframe
df_national['ContagionRate'] = pd.Series(output)
df_national.to_csv('contagiofrancia.csv')
dummy = np.zeros(60)
plt.figure(figsize= (17,10))
plt.subplot(211)
plt.plot(df_national['date'],df_national['cases'], color = 'g') #trend cases
'''
plt.plot(timerange,dummy, ':', color = 'w') 
plt.title('Cases over time')
plt.ylabel('number of cases')
plt.xticks(df_national['date']," ")
plt.subplot(212)
plt.plot(df_national['date'],df_national['ContagionRate'], color = 'r', label = 'new cases') #trend daily cases
plt.plot(timerange,y_e, '--', color = 'orange', label = 'gaussian fit') 
plt.title('Spread rate over time')
plt.ylabel('Rate (new cases per day)')
plt.legend()
plt.xticks(rotation=90)

plt.suptitle('Virus spread over time - France')
'''
plt.show()