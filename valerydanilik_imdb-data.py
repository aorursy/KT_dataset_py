import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

import matplotlib.pyplot as plt

import seaborn as sns



DATA_PATH = "../input/movie_metadata.csv"
imdb_data = pd.read_csv(DATA_PATH)
imdb_data.head(10)

budget_score = imdb_data.loc[:,["budget","imdb_score"]].dropna()
budget_score.head()
budget_score.ix[:,"imdb_score"]
sns.set(style="white", palette="muted", color_codes=True)

sns.distplot(budget_score.imdb_score, kde=True)

sns.despine(left=True, bottom=True)
plt.hist(budget_score.budget)
imdb_data.describe()