import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

pokemon=pd.read_csv("../input/Pokemon.csv")
pokemon.head()
pokemon.shape
sns.distplot(pokemon['Total'])

sns.plt.show()

#peak at 300 and 500
pokemon['Type 1'].value_counts()
len(pokemon['Type 1'].value_counts())
pokemon['Type 2'].value_counts()
pokemon['Legendary'].value_counts()
pokemon['Generation'].value_counts()
#explore based on type

g1=pokemon.groupby("Type 1")
g1["Total","HP","Attack","Defense","Sp. Atk","Sp. Def","Speed"].mean()
g2=pokemon.groupby(['Type 1','Type 2'])

g2["Total","HP","Attack","Defense","Sp. Atk","Sp. Def","Speed"].aggregate([np.mean, np.median])
g2.size()
#legendary pokemon

lg=pokemon.loc[pokemon["Legendary"] == True]
lg.head()
lg['Generation'].value_counts()
lg['Type 1'].value_counts()
lg3=pokemon.loc[(pokemon["Legendary"] == True) & (pokemon["Generation"] == 3)]

lg5=pokemon.loc[(pokemon["Legendary"] == True) & (pokemon["Generation"] == 5)]
lg31=lg3.groupby("Type 1")

lg51=lg5.groupby("Type 1")

lg31.mean()
lg51.mean()

#difference on type btw generation 1 & 6
sns.jointplot(x="HP", y="Attack", data=pokemon,kind="reg")

sns.plt.show()
pokemon = pokemon.drop(['#'],1)
pkm=pokemon.loc[:,["Name","Total","HP","Attack","Defense","Sp. Atk","Sp. Def","Speed"]]
pkm.head()
sns.boxplot(data=pkm)

sns.plt.show()
# advantages of legendary

sns.boxplot(data=lg)

sns.plt.show()
#relationships among numerical features for all kinds

sns.pairplot(pkm, vars=['HP', 'Attack','Defense','Sp. Atk','Sp. Def','Speed'])

sns.plt.show()