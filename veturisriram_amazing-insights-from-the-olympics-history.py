import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/athlete_events.csv')
regions = pd.read_csv('../input/noc_regions.csv')
data.head()
data.describe()
data.info()
regions.head()
merged = pd.merge(data, regions, on='NOC', how='left')
merged.head()
goldMedals = merged[merged.Medal == 'Gold']
goldMedals.head()
goldMedals.isnull().any()
goldMedals = goldMedals[np.isfinite(goldMedals['Age'])]
plt.figure(figsize=(20, 10))
plt.tight_layout()
p1 = sns.countplot(goldMedals['Age'])
p1.set_xticklabels(p1.get_xticklabels(),rotation=45)
plt.title('Distribution of Gold medals with Athlete Age!')
plt.figure(figsize=(20, 10))
sns.countplot(goldMedals['Height'])
plt.tight_layout()
plt.title('Distribution of Heights')
masterDisciplines = goldMedals['Sport'][goldMedals['Age'] > 50]
plt.figure(figsize=(20, 10))
plt.tight_layout()
p2 = sns.countplot(masterDisciplines)
plt.title('Sports for athletes over age 50')
women = merged[(merged.Sex == 'F') & (merged.Season == 'Summer')]
women.head()
women.shape
sns.set(style = 'darkgrid')
plt.figure(figsize=(20,10))
sns.countplot(women['Year'])
plt.title('Women Medals Evolution')
goldMedals.region.value_counts().head()
goldMedalsUSA = goldMedals.loc[goldMedals['NOC'] == 'USA']
goldMedalsUSA.Event.value_counts().head(20)
basketballGoldUSA = goldMedalsUSA.loc[(goldMedalsUSA['Sport'] == 'Basketball') & (goldMedalsUSA['Sex'] == 'M')].sort_values(['Year'])
basketballGoldUSA.head(15)
groupedBasketUSA = basketballGoldUSA.groupby(['Year']).first()
groupedBasketUSA.head()
notNullMedals = goldMedals[(goldMedals['Height'].notnull()) & (goldMedals['Weight'].notnull())]
notNullMedals.head()
plt.figure(figsize=(20,20))
ax = plt.scatter(x='Height', y='Weight', data=notNullMedals)
plt.title('Height vs Weight of Olympics Medalists')
notNullMedals.loc[notNullMedals['Weight'] > 150].head()
notNullMedals.loc[notNullMedals['Height'] > 215].head()
notNullMedals.loc[notNullMedals['Height'] < 140].head()
notNullMedals.loc[notNullMedals['Weight'] < 30].head()
womenData = merged[(merged.Sex == "F") & (merged.Sport == "Athletics")]
plt.figure(figsize=(20,10))
sns.countplot(womenData['Height'])
plt.tight_layout()
plt.title("Women Height Distribution in Athletics")
indianData = merged[(merged.Team == "India") & (merged.Season == "Summer")]
plt.figure(figsize=(20,10))
sns.countplot(indianData['Year'])
plt.tight_layout()
plt.title('Indian Gold Medal Events')
indiansInOlympics = merged[merged.Team == "India"]
indiansInOlympics.head()
plt.figure(figsize=(20,10))
sns.countplot(indiansInOlympics['Medal'])
plt.tight_layout()
plt.title('Medal Ratio of Indian Olympians')
plt.figure(figsize=(20,10))
sns.countplot(indiansInOlympics['Year'])
plt.tight_layout()
plt.title('Medal evolution of Indian Olympians')
plt.figure(figsize=(20,10))
goldIndia = indiansInOlympics[indiansInOlympics.Medal == "Gold"]
sns.countplot(goldIndia['Sex'])
plt.tight_layout()
plt.title('Medal Ratio of Indian Olympians')