import pandas as pd

import seaborn as sns
data = pd.read_csv("../input/winemag-data_first150k.csv")
data.head()
sns.countplot(data['points'])
data[data['price']<200]['price'].value_counts().sort_index().plot.line()
#sns.kdeplot(data[data['price']<200]['price'])



sns.kdeplot(data.query('price<200').price)
sns.kdeplot(data[data['price']<200][['price','points']].sample(5000))
sns.distplot(data['points'],bins=10,kde=False)
sns.jointplot(x='price', y='points', data = data[data['price']<200])
sns.jointplot(x='price',y='points',data = data[data['price']<200], kind='hex',gridsize=20)
data.head()
df = data[data['variety'].isin(data['variety'].value_counts().head(5).keys())]

sns.boxplot(x='variety',y='points', data = df)





df = data[data['variety'].isin(data['variety'].value_counts().head().keys())]



sns.violinplot(x='variety' , y='points' , data=df)
data.head()
data[data['province'].isin(data['province'].value_counts().head(3).keys())]
sns.kdeplot(data[data['province']=='California']['points'])

sns.kdeplot(data[data['province']=='Tuscany']['points'])
data.head()



df = data[data['province'].isin(['California','Tuscany','Oregon'])]



g = sns.FacetGrid(df, col='province')



g.map(sns.kdeplot , 'points')
df = data[data['province'].isin(data['province'].value_counts().head(20).keys())]



g = sns.FacetGrid(df , col='province', col_wrap=5)



g.map(sns.kdeplot , 'points')
data.head()
df = data[data['country'].isin(['US','Italy','France']) & data['variety'].isin(['Chardonnay','Pinot Noir','Cabernet Sauvignon'])]



g = sns.FacetGrid(df , col='country' , row='variety')

g.map(sns.violinplot , 'points')
data.head()
sns.pairplot(data[['points','price']])
data.head()
sns.lmplot(x='price',y='points',hue='variety',markers=['o','^','+'], data = data[data['variety'].isin(['Tinta de Toro','Pinot Noir','Sauvignon Blanc'])], fit_reg=False)
df = data[data['variety'].isin(['Port','Barbera']) & (data['price']<15)]



sns.boxplot(x='price' , y='points' , hue='variety' , data=df)