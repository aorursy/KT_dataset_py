from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



euro = pd.read_csv('../input/euromillions-draw-history.csv')

lotto= pd.read_csv('../input/lotto-draw-history.csv')

thunder = pd.read_csv('../input/thunderball-draw-history.csv')
thunder.head()
thunder.describe()
plt.matshow(thunder.corr())

plt.colorbar()

plt.show()
plt.scatter(thunder['T1'], thunder['TB'])
sns.regplot(x=thunder['T1'], y=thunder['TB'])
sns.lineplot(x='T1', y='TB', data=thunder)
plt.style.use('fast')

sns.jointplot(x = 'T1', y = 'TB', data = thunder)

plt.show()
q1 = sns.boxenplot(x = thunder['T1'], y = thunder['TB'], palette = 'rocket')
euro.head()
euro.describe()
plt.matshow(euro.corr())

plt.colorbar()

plt.show()
plt.scatter(euro['LS1'], euro['LS2'])
sns.regplot(x=euro['LS1'], y=euro['LS2'])
sns.lineplot(x='LS1', y='LS2', data=euro)
plt.style.use('fast')

sns.jointplot(x = 'LS1', y = 'LS2', data = euro)

plt.show()
q2 = sns.boxenplot(x = euro['LS1'], y = euro['LS2'], palette = 'rocket')
lotto.head()
lotto.describe()
plt.matshow(lotto.corr())

plt.colorbar()

plt.show()
plt.scatter(lotto['T1'], lotto['Bonus Ball'])
sns.regplot(x=lotto['T1'], y=lotto['Bonus Ball'])
sns.lineplot(x='T1', y='Bonus Ball', data=lotto)
plt.style.use('fast')

sns.jointplot(x = 'T1', y = 'Bonus Ball', data = lotto)

plt.show()
q2 = sns.boxenplot(x = lotto['T1'], y = lotto['Bonus Ball'], palette = 'rocket')