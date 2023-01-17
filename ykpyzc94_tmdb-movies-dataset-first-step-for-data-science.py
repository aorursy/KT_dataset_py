# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt  #for plot 

import seaborn as sns            #for heatmap
data = pd.read_csv("../input/tmdb_5000_movies.csv")   #using for read dataset
data.info()
data.head()
data.corr()

f,ax = plt.subplots(figsize=(10,10))

sns.heatmap(data.corr(), annot = True, linewidths = .1, fmt = '.2f', ax = ax)

plt.show()
data.columns
plt.subplot()

data.budget.plot(kind = 'line', color = 'r', label = 'Budget', linewidth = 1, alpha = 0.7, grid = True, linestyle = '-', figsize = (20,10))

data.revenue.plot(kind = 'line', color = 'b', label = 'Revenue', linewidth = 1, alpha = 0.7, grid = True, linestyle = '-', figsize = (20,10))

plt.title('Budget - Revenue Line Plot')

plt.xlabel('Movies ID')

plt.ylabel('Value')

plt.legend(loc = 'upper right')

plt.show()
data.popularity.plot(kind = 'line', color = 'g', label = 'Popularity', linewidth = 1, alpha = 0.7, grid = True, linestyle = '-', figsize = (20,10))

plt.title('Popularity Line Plot')

plt.xlabel('Movies ID')

plt.ylabel('Value')

plt.legend(loc = 'upper right')

plt.show()
data.plot(kind = 'scatter', x = 'popularity',  y = 'budget', alpha = 0.5, figsize = (20,10))

plt.xlabel('Popularity')

plt.ylabel('Budget')

plt.title(' Popularity - Budget Correlation')

plt.show()

data_filter_p = data['popularity'] < 200

data_filter_popularity = data[data_filter_p]

data_filter_popularity.plot(kind = 'scatter', x = 'popularity', y = 'budget', alpha = 0.5, figsize = (20,10))

plt.xlabel('Popularity')

plt.ylabel('Budget')

plt.title(' Popularity - Budget Correlation')

plt.show()
data_filter_p2 = np.logical_and(data['popularity'] > 20, data['popularity'] < 100)

data_filter_popularity_x2 = data[data_filter_p2]

data_filter_popularity_x2.plot(kind = 'scatter', x = 'popularity', y = 'budget', alpha = 0.5, figsize = (20,10))

plt.xlabel('Popularity')

plt.ylabel('Budget')

plt.title(' Popularity - Budget Correlation')

plt.show()
data.plot(kind = 'scatter', x = 'popularity', y = 'revenue', alpha = 0.5, color = 'r', figsize = (20, 10))

plt.xlabel('Popularity')

plt.ylabel('Revenue')

plt.title(' Popularity - Revenue Correlation')

plt.show()
data_filter_popularity.plot(kind = 'scatter', x = 'popularity', y = 'revenue', alpha = 0.5, color = 'r', figsize = (20,10))

plt.xlabel('Popularity')

plt.ylabel('Revenue')

plt.title(' Popularity - Revenue Correlation')

plt.show()
data_filter_popularity_x2.plot(kind = 'scatter', x = 'popularity', y = 'revenue', alpha = 0.5, color = 'r', figsize = (20,10))

plt.xlabel('Popularity')

plt.ylabel('Revenue')

plt.title(' Popularity - Revenue Correlation')

plt.show()
data.plot(kind = 'scatter', x = 'budget', y = 'revenue', alpha = 0.5, color = 'g', figsize = (20,10))

plt.xlabel('Budget')

plt.ylabel('Revenue')

plt.title(' Budget - Revenue Correlation')

plt.show()
data_filter_b = data['budget'] < 150000000

data_filter_budget = data[data_filter_b]

data_filter_budget.plot(kind = 'scatter', x = 'budget', y = 'revenue', alpha = 0.5, color = 'g', figsize = (20,10))

plt.xlabel('Budget')

plt.ylabel('Revenue')

plt.title(' Budget - Revenue Correlation')

plt.show()
data.original_language.unique()