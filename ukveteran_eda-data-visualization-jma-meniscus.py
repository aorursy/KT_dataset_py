import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns; sns.set(color_codes=True)  # visualization tool





from sklearn.linear_model import LinearRegression



import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv("../input/meniscus-repair-methods/Meniscus.csv")
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
sns.pairplot(df.iloc[:,0:8],hue="Method")

plt.show()
fig=plt.figure(figsize=(20,15))

ax=fig.gca()

df.hist(ax=ax)

plt.show()
fig=plt.figure(figsize=(20,10))

sns.boxplot(data = df,notch = True,linewidth = 2.5, width = 0.50)

plt.show()
fig,axes=plt.subplots(nrows=2,ncols=1)



df.plot(kind='hist',y='Displacement',bins=50,range=(0,100),density=True,ax=axes[0])

df.plot(kind='hist',y='Displacement',bins=50,range=(0,100),density=True,ax=axes[1],cumulative=True)

plt.show()
print(df['Displacement'].value_counts(dropna=False))
sns.barplot(x='Displacement',y='Stiffness',data=df)

plt.show()
sns.jointplot(x=df.Displacement, y=df.Stiffness, data=df, kind="kde");
sns.swarmplot(x = 'Displacement', y = 'Stiffness', data = df)

plt.show()
df.boxplot(column='Displacement', by='Stiffness')

plt.show()
with sns.axes_style("white"):

    sns.jointplot(x=df.Displacement, y=df.Stiffness, kind="hex", color="k");
new_df=df.iloc[:,[0,3,4]]

new_df.head()
color_list = ['red' if i==1 else 'green' for i in new_df.loc[:,'Displacement']]

pd.plotting.scatter_matrix(new_df.loc[:, new_df.columns != 'Displacement'],

                                       c=color_list,

                                       figsize= [15,15],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s = 200,

                                       marker = '*',

                                       edgecolor= "black")

plt.show()
color_list = ['red' if i==1 else 'green' for i in new_df.loc[:,'Stiffness']]

pd.plotting.scatter_matrix(new_df.loc[:, new_df.columns != 'Stiffness'],

                                       c=color_list,

                                       figsize= [15,15],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s = 200,

                                       marker = '*',

                                       edgecolor= "black")

plt.show()