import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



data = pd.read_csv('../input/movie_metadata.csv')
data.head()
data.shape
data.color.value_counts()
sns.boxplot(x='color', y='imdb_score', data=data)
sns.boxplot(x='color', y='duration', data=data)
f, axes = plt.subplots(2, 1, sharex=True)

sns.violinplot(x='title_year', y='color', data=data, ax=axes[0])

sns.distplot(data.title_year[-pd.isnull(data.title_year)], kde=False, ax=axes[1])
pair_data = data[['duration', 'gross', 'budget', 'title_year', 'imdb_score', 'aspect_ratio']]

pair_data = pair_data.dropna()
sns.distplot(data.budget[data.budget < data.budget.mean() + data.budget.std() * 2])
g = sns.PairGrid(pair_data)

g.map_offdiag(plt.scatter)

g.map_diag(plt.hist)
print(data.budget.median())

print(data.budget.mean() - data.budget.median())
sns.distplot(data.budget[data.budget < data.budget.mean() + data.budget.std() * 3])
data['director_name_length'] = data.director_name.map(lambda x: len(str(x)))

data.director_name_length.head()
sns.distplot(data.director_name_length)
sns.regplot(x='director_name_length', y='imdb_score', data=data, lowess=True)
sns.violinplot(y='content_rating', x='imdb_score', data=data)
r_data = data[data.content_rating == 'R']

sns.regplot(x='title_year', y='imdb_score', data=r_data, lowess=True)