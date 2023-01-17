import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns; sns.set(color_codes=True)  # visualization tool





from sklearn.linear_model import LinearRegression



import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv("../input/wages-data/Griliches.csv")
type(df)
df.head()
df.tail()
df.shape
df.describe()
df.info()
df.columns
df.dtypes
df.corr()
df.plot(subplots=True,figsize=(18,18))

plt.show()
plt.figure(figsize=(15,10))

sns.heatmap(df.iloc[:,0:15].corr(), annot=True,fmt=".0%")

plt.show()
sns.pairplot(df.iloc[:,0:8],hue="rns")

plt.show()
fig=plt.figure(figsize=(20,15))

ax=fig.gca()

df.hist(ax=ax)

plt.show()
fig=plt.figure(figsize=(20,10))

sns.boxplot(data = df,notch = True,linewidth = 2.5, width = 0.50)

plt.show()
fig,axes=plt.subplots(nrows=2,ncols=1)



df.plot(kind='hist',y='age',bins=50,range=(0,100),density=True,ax=axes[0])

df.plot(kind='hist',y='age',bins=50,range=(0,100),density=True,ax=axes[1],cumulative=True)

plt.show()
print(df['tenure'].value_counts(dropna=False))
sns.barplot(x='age',y='tenure',data=df)

plt.show()
sns.jointplot(x=df.age, y=df.tenure, data=df, kind="kde");
sns.swarmplot(x = 'age', y = 'tenure', data = df)

plt.show()
fig=plt.figure(figsize=(20,5))

sns.violinplot(x ='age', y = 'tenure', data = df)

plt.show()
df.boxplot(column='age', by='tenure')

plt.show()
with sns.axes_style("white"):

    sns.jointplot(x=df.age, y=df.tenure, kind="hex", color="k");
new_df=df.iloc[:,[0,7,8,9,10,11,12,13,14]]

new_df.head()
color_list = ['red' if i==1 else 'green' for i in new_df.loc[:,'age']]

pd.plotting.scatter_matrix(new_df.loc[:, new_df.columns != 'age'],

                                       c=color_list,

                                       figsize= [15,15],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s = 200,

                                       marker = '*',

                                       edgecolor= "black")

plt.show()