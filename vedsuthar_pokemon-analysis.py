import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

pokedex = pd.read_csv('../input/complete-pokemon-dataset-updated-090420/pokedex_(Update_05.20).csv')

pokedex = pokedex.drop(['Unnamed: 0'], axis=1)

pokedex.info()
pokedex.head()
genCount=pokedex.groupby("generation").size().to_numpy()

plt.plot(np.arange(1,9),genCount)

plt.xlabel('Generation')

plt.ylabel('Count')

print('Raw:',genCount)

print('Mean:',genCount.mean())

      
ax = sns.countplot(x="type_1", data=pokedex);

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

ax.set(xlabel='Type 1', ylabel='Count', title='First type');
null_type_2=pokedex.type_2.isnull().sum()

notnull_type_2=pokedex.type_2.size-null_type_2

plt.pie([null_type_2,notnull_type_2],labels=['Have type 2','Does not have type 2'],autopct='%1.1f%%')
bx = sns.countplot(x="type_2", data=pokedex);

bx.set_xticklabels(bx.get_xticklabels(), rotation=90, ha="right")

bx.set(xlabel='Type 2', ylabel='Count', title='Second type');


sns.heatmap(pd.crosstab(pokedex.type_1,pokedex.type_2),annot=True)

sns.distplot(pokedex['height_m'])

print('Max height:',pokedex.loc[pokedex['height_m'].idxmax()]['name'])

print('Min height:',pokedex.loc[pokedex['height_m'].idxmin()]['name'])
sns.distplot(pokedex['weight_kg'])

print('Max weight:',pokedex.loc[pokedex['weight_kg'].idxmax()]['name'])

print('Min weight:',pokedex.loc[pokedex['weight_kg'].idxmin()]['name'])
data=pokedex[['weight_kg','height_m']].dropna()

weight=np.log(data['weight_kg'].to_numpy())

height=np.log(data['height_m'].to_numpy())

sns.kdeplot(weight,height)

plt.xlabel('Log of weight')

plt.ylabel('Log of height')
sx = sns.countplot(x="status", data=pokedex);

sx.set_xticklabels(sx.get_xticklabels(), rotation=90, ha="right")

sx.set(xlabel='Status', ylabel='Count', title='Status');

print('Raw:',pokedex.groupby('status').size())
STATS_CATEGORIES = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]

pokedex[STATS_CATEGORIES].describe()



for each in STATS_CATEGORIES:

    sns.distplot(pokedex[each])

plt.legend(STATS_CATEGORIES)

plt.xlabel('Refer Legend')
sns.distplot(pokedex['total_points'])
for colm in STATS_CATEGORIES:

    print('Max',colm,':',pokedex.loc[pokedex[colm].idxmax()]['name'])

    print('Min',colm,':',pokedex.loc[pokedex[colm].idxmin()]['name'])
for each in STATS_CATEGORIES:

    print('----',each,'----')

    print(pokedex.groupby('status')[each].mean())



print('---- Total Points ----')

print(pokedex.groupby('status')['total_points'].mean())
sns.heatmap(pd.pivot_table(pokedex,values=STATS_CATEGORIES,columns='status'),annot=True,fmt='.1f')
sns.heatmap(pokedex.groupby('type_1')[STATS_CATEGORIES].mean(),annot=True)

null_type_2=pokedex.percentage_male.isnull().sum()

notnull_type_2=pokedex.percentage_male.size-null_type_2

plt.pie([notnull_type_2,null_type_2],labels=['Have gender','Does not have gender'],autopct='%1.1f%%')
pokedex.groupby('status').percentage_male.count().plot.bar()

plt.ylabel('Has gender')
plt.bar(['Normal','Sub Legendary'],[pokedex[pokedex['status'].isin(['Normal'])].percentage_male.mean(),pokedex[pokedex['status'].isin(['Sub Legendary'])].percentage_male.mean()])

plt.xlabel('Status')

plt.ylabel('Percentage Male')
pokedex.groupby('status').catch_rate.describe()
pokedex.groupby('status').catch_rate.mean().plot.bar()

plt.ylabel('Mean Catch Rate')
pokedex.groupby(['status','growth_rate']).pokedex_number.count()
sns.heatmap(pd.crosstab(pokedex.status,pokedex.growth_rate).apply(lambda r: r/r.sum()*100, axis=1),annot=True)