# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/recorded-crime-data-at-police-force-area-level/rec-crime-pfa.csv')
df.isnull().sum()
df.nunique()
df.head()
df['12 months ending']=pd.to_datetime(df['12 months ending'])

df['Year']=df['12 months ending'].dt.year

df['Month']=df['12 months ending'].dt.month

df['Date']=df['12 months ending'].dt.day

df.drop(['12 months ending'],axis=1,inplace=True)
df
df['Month'].value_counts()
df['Offence'].unique()
df.loc[df['Offence']=='Domestic burglary']['Offence']='Burglary'
df.loc[df['Offence'] == 'Domestic burglary', 'Offence'] = 'Burglary'

df.loc[df['Offence'] == 'Non-domestic burglary', 'Offence'] = 'Burglary'

df.loc[df['Offence'] == 'Non-residential burglary', 'Offence'] = 'Burglary'

df.loc[df['Offence'] == 'Residential burglary', 'Offence'] = 'Burglary'

#df.loc[df['Offence'] == 'Domestic burglary', 'Offence'] = 'Burglary'

#df.loc[df['Offence'] == 'Domestic burglary', 'Offence'] = 'Burglary'



df.loc[df['Offence'] == 'Non-domestic burglary', 'Offence'] = 'Burglary'

df.loc[df['Offence'] == 'Non-residential burglary', 'Offence'] = 'Burglary'

df.loc[df['Offence'] == 'Residential burglary', 'Offence'] = 'Burglary'



df.loc[df['Offence'] == 'Bicycle theft', 'Offence'] = 'Theft'

df.loc[df['Offence'] == 'Shoplifting', 'Offence'] = 'Theft'

df.loc[df['Offence'] == 'Theft from the person', 'Offence'] = 'Theft'

df.loc[df['Offence'] == 'All other theft offences', 'Offence'] = 'Theft'



df.loc[df['Offence'] == 'Violence with injury', 'Offence'] = 'Violence'

df.loc[df['Offence'] == 'Violence without injury', 'Offence'] = 'Violence'

df.head(3)
plt.figure(figsize=(20,20))

sns.barplot(x='Year', y='Rolling year total number of offences', data=df)

sns.barplot(df['Month'],df['Rolling year total number of offences'])
plt.figure(figsize=(20,20))

z=sns.barplot(x='Region', y='Rolling year total number of offences', data=df)

plt.xticks(rotation=45)
df2=df[df['Year']>2014]

z=df2.sort_values('Rolling year total number of offences',ascending=False)
z=df[df['Region']=="East"]
z
plt.figure(figsize=(20,20))

sns.barplot(df['Offence'],df['Rolling year total number of offences'])

plt.xticks(rotation=45)
plt.figure(figsize=(20,20))

sns.barplot(df['Year'],df['Rolling year total number of offences'])

plt.xticks(rotation=45)
z.groupby('Offence').sum()
plt.figure(figsize=(20,20))

sns.lineplot(df['Offence'],df['Rolling year total number of offences'])

plt.xticks(rotation=75)