import pandas as pd

import numpy as np

import seaborn as sns 

import matplotlib.pyplot as plt
df = pd.read_csv("../input/training-data/train.csv.csv")

df
df.head()
df.shape
df.dtypes
type(df)
df.columns
df.info()
df.describe()
df['Name'].value_counts()
df['Name'].value_counts('normlise=True')
df.loc[0:5,'Name':'Age']
df.iloc[0:5,0:5]
df[df['Age']== df[df['Sex'] =='male']['Age'].max()]['Name']
df.sort_values(by='Ticket',ascending='True')
df.sort_values(by='Age',ascending='True').head()
df.sort_values(by=['Name','Age'],ascending=[True,False]).head()
df.groupby(by='Survived')['Age'].describe()
df.groupby(by='Embarked')['Age'].agg(np.mean)
pd.crosstab(df['Sex'],df['Survived'], normalize=True)
df.pivot_table(['Age', 'Fare'],['Survived'], aggfunc='mean')
sns.lineplot(x='Age',y='Fare',data=df)
sns.factorplot(x='Survived', data=df, kind='count')
sns.factorplot(x='Survived', data=df, kind='count', hue='Sex')
df['Age'].hist(bins=8)
as_fig = sns.FacetGrid(df,hue='Sex',aspect=5)

as_fig.map(sns.kdeplot,'Age',shade=True)

oldest = df['Age'].max()

as_fig.set(xlim=(0,oldest))

as_fig.add_legend()
df.plot.scatter(x='Age',y='Fare')
df.plot.scatter(x='Age',y='Survived')
import matplotlib.pyplot as plt

sizes= df['Survived'].value_counts()

fig1,ax1 = plt.subplots()

ax1.pie(sizes,labels=['Not Survived', 'Survived'],autopct='%1.1f%%',shadow=True)

plt.show()