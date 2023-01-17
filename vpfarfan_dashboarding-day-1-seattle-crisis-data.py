# Load libraries and import data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

df = pd.read_csv('../input/crisis-data.csv')
print('Dimensions: ', df.shape)
print('Unique IDs: ', df['Template ID'].nunique())
print('Data from ', min(df['Reported Date']), ' to ', max(df['Reported Date']))
df.head()
# Delete rows with bad date data
df = df.loc[df['Reported Date'] != '1900-01-01',:]
# Set column types
df['Reported Date'] = pd.to_datetime(df['Reported Date'])
df['Precinct'] = pd.Categorical(df['Precinct'])
df['Date'] = pd.to_datetime(df['Reported Date']) - pd.to_timedelta(7, unit='d')
fig, ax = plt.subplots(figsize=(14, 8))
df.groupby([pd.Grouper(key='Date', freq='W-MON'), 'Precinct'])['Template ID'].nunique().reset_index(level=1).last("5M").reset_index().pivot(index='Date', columns='Precinct', values= 'Template ID').plot(ax=ax)
plt.title('Events by week by precinct')
plt.show()
fig, ax = plt.subplots(figsize=(14, 10))
(df[['Template ID', 'Disposition']].groupby(['Disposition'])['Template ID'].nunique().sort_values()).plot.barh(ax=ax)
plt.title('Events by Disposition')
plt.show()
