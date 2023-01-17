import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd

combats = pd.read_csv("../input/pokemon-challenge/combats.csv")

pokemon = pd.read_csv("../input/pokemon-challenge/pokemon.csv")

tests = pd.read_csv("../input/pokemon-challenge/tests.csv")
pokemon.info()
pokemon.head(10)
pokemon.corr()
f,ax = plt.subplots(figsize=(18,18))

sns.heatmap(pokemon.corr(), annot=True , linewidths=.5 , fmt='.1f', ax = ax)

plt.show()
pokemon.columns

pokemon.Speed.plot(kind ='line', color ='g',label ='Speed',linewidth=1,alpha =0.7,grid =True,linestyle =':')

pokemon.Defense.plot(color='r',label='Defense',linewidth=1,alpha=0.5,grid =True,linestyle ='-')

plt.legend(loc= 'upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()

pokemon.plot(kind = 'scatter', x = 'Attack', y='Defense', alpha = .5 , color='blue')

plt.xlabel('Attack')

plt.ylabel('Defense')

plt.title('Attack & Defense Scatter Plot')

plt.show()
pokemon.Attack.plot(kind='hist', bins = 50 , figsize= (15,15))

plt.show()