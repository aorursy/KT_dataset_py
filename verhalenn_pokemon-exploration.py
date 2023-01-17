import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv('../input/Pokemon.csv', index_col=0)

print(data.shape)

data.head()
data.dtypes
scatter_data = data[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Type 1']]

g = sns.PairGrid(scatter_data, hue='Type 1')

g.map_diag(plt.hist)

g.map_offdiag(plt.scatter)
scatter_data = data[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']]

g = sns.PairGrid(scatter_data, hue='Generation')

g.map_diag(plt.hist)

g.map_offdiag(plt.scatter)
data.Generation.value_counts()
mean_data = data.groupby('Generation').mean()

mean_data
for column in mean_data.columns[1:-1]:

    plt.plot(mean_data.index, mean_data[column])

    

plt.xlabel('Generation')

plt.ylabel('Rating')

plt.legend()
plt.plot(mean_data.index, mean_data.Total)

plt.xlabel('Generation')

plt.ylabel('Total Rating')
print(data['Type 1'].value_counts())

print(data['Type 2'].value_counts())
pd.crosstab(data.Generation, data['Type 1'], normalize='index') * 100
pd.crosstab(data.Generation, data['Type 2'], normalize='index') * 100