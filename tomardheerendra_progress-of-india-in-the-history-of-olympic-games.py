import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os 
print(os.listdir("../input"))
data = pd.read_csv('../input/athlete_events.csv')
regions = pd.read_csv('../input/noc_regions.csv')
data.head(5)
data.describe()
data.info()
regions.head()
merged = pd.merge(data, regions, on='NOC', how='left')
merged.head()
dataIndia = merged[merged.NOC == 'IND']
dataIndia.head(5)
dataIndia.isnull().any()
dataIndia = dataIndia[dataIndia.Medal.notnull()]
dataIndia.head()
plt.figure(figsize=(12, 6))
plt.tight_layout()
sns.countplot(x='Year', hue='Medal', data= dataIndia)
plt.title('Distribution of Medals')
plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.countplot(x='Sport', data= dataIndia)
plt.title('Distribution of Medals')
hockeyPlayersMedal = dataIndia.loc[(dataIndia['Sport'] == 'Hockey')].sort_values(['Year'])
hockeyPlayersMedal.head(30)
hockeyTeamMedal = hockeyPlayersMedal.groupby(['Year']).first()
hockeyTeamMedal.head()
hockeyTeamMedal['ID'].count()
plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.countplot(x='Year', hue='Sport', data= dataIndia)
plt.title('Distribution of Medals by sports')
womenInOlympics = merged[(merged.Sex == 'F') & (merged.NOC == 'IND')]
womenInOlympics.head()
sns.set(style="darkgrid")
plt.figure(figsize=(20, 10))
sns.countplot(x='Year', data=womenInOlympics)
plt.title('Women participation per edition of the Games')
womenInOlympics = dataIndia[dataIndia.Sex == 'F']
womenInOlympics.head()
sns.set(style="darkgrid")
plt.figure(figsize=(20, 10))
sns.countplot(x='Year', hue='Medal', data=womenInOlympics)
plt.title('Women Medals per edition of the Games')
dataIndia.Medal.value_counts()
plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.countplot(x='Medal', hue='Year', data= dataIndia)
plt.title('Distribution of Medals by year')
