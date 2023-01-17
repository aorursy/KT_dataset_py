import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from pylab import rcParams



%matplotlib inline

rcParams['figure.figsize'] = 12, 6
# lOAD the CSV File into the Memory

df = pd.read_csv('../input/Crime_Data_from_2010_to_Present.csv')
df.info()
# First 5 Rows of the DataFrame

df.head()
# Last 5 Rows

df.tail()
# Select the Columns with Victim's Age, Sex and Descent.

victim = df[['Victim Age', 'Victim Sex', 'Victim Descent']]
# Checking whether we have NULL Values.

victim.isnull().any()
# Histogram using Matplotlib.



bins = np.arange(victim['Victim Age'].min(), victim['Victim Age'].max(), 5)

plt.hist(victim['Victim Age'].dropna(), bins=bins, alpha=1, edgecolor='black')
# Distplot using Seaborn

sns.distplot(victim['Victim Age'].dropna())
# Seaborns Count Plot

plt.title('Number of Victims as per Sex')

sns.countplot('Victim Sex', data=victim)
# Number of Victims according to Sex of the Victim b/w the age group 20 and 35 years

sns.countplot('Victim Sex', data=victim[(victim['Victim Age'] >= 20) & (victim['Victim Age'] <= 35)])
# Number of Victims according to Sex of the Victim b/w the age group 10 and 15 years

sns.countplot('Victim Sex', data=victim[(victim['Victim Age'] >= 10) & (victim['Victim Age'] <= 15)])
victim['Victim Descent'].unique()
sns.countplot('Victim Descent', data=victim)
# Area ID, Area Name and Reporting District.

area = df[['Area ID', 'Area Name', 'Reporting District']]
area.isnull().any()
# Which area have more Crime.

plt.title('Number of Crimes in Different Area')

sns.countplot(x='Area ID', data=area)
# Crime Code and it's Description

crime = df[['Crime Code', 'Crime Code Description']]
# Checking NULL Values.

crime.isnull().any()
crime['Crime Code Description'].value_counts()[:10]
plt.title('Number of Crimes according to Crime Description')

crime['Crime Code Description'].value_counts()[:10].plot(kind='bar')
plt.title('Number of Crimes according to Crime Code')

crime['Crime Code'].value_counts()[:10].plot(kind='bar')
area_crime = pd.concat(objs=[area, crime], axis=1)
print ("Number of Crimes in Area ID 12 i.e 77th Street", len(area_crime[(area_crime['Area ID'] == 12)]))

print ("Number of Crimes in Area ID 3 i.e Southwest", len(area_crime[(area_crime['Area ID'] == 3)]))
# Select the Area with the ID 12 and 3 Respectively.

area_crime_12 = area_crime[(area_crime['Area ID'] == 12)]

area_crime_3 = area_crime[(area_crime['Area ID'] == 3)]
area_crime_12['Crime Code Description'].value_counts()[:10]
area_crime_3['Crime Code Description'].value_counts()[:10]