import pandas as pd

import seaborn as sns
data = pd.read_csv("../input/pokemon.csv")



data.head()
data['type1'].value_counts().plot.bar()
data['hp'].value_counts().sort_index().plot.line()
data['weight_kg'].value_counts().sort_index().plot.hist()

data.head()
data.plot.hexbin(x='sp_attack',y='sp_defense',gridsize=12)
data.plot.scatter(x='sp_attack',y='sp_defense')
data.groupby(['generation','is_legendary']).mean()[['sp_attack','sp_defense']]
data.groupby(['generation','is_legendary']).mean()[['sp_attack','sp_defense']].plot.bar(stacked=True)
data
data.groupby('generation').mean()[['hp','attack','defense','sp_attack','sp_defense','speed']]
data.groupby('generation').mean()[['hp','attack','defense','sp_attack','sp_defense','speed']].plot.line()
data.head()
sns.countplot(data['generation'])
sns.distplot(data['hp'] , kde=False)
sns.jointplot(x='attack' , y='defense' , data=data)
sns.jointplot(data['attack'] , data['defense']  , kind='hex')
sns.kdeplot(data['hp'] , data['attack'])
sns.boxplot(x='is_legendary' , y='attack' , data=data)
sns.violinplot(data['is_legendary'] , data['attack'])