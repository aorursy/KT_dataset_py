# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/winemag-data-130k-v2.csv")
data.info()
data.describe()
data.corr()
# Correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()
data.head(10)
data.columns
# Line plot
data.points.plot(kind = "line", color = "b", label = "points",linewidth = 1, alpha = 0.5, grid = True, linestyle = ":")
data.price.plot(color = "y", label = "price",linewidth = 1, alpha = 0.5, grid = True, linestyle = "-.")
plt.legend()
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Point - Price graph")
plt.show()
# Scatter Plot
data.plot(kind  = "scatter", x = "price", y = "points", alpha = 0.4, color = "blue")
plt.title("Point - Price graph")
plt.xlabel("price")
plt.ylabel("points")
plt.show()
# Histogram
data.points.plot(kind = "hist", bins = 20, figsize=(12,12))
plt.show()
# filtering
data[(data['points'] > 90) & (data['price'] > 2000)]
data[(data['country'] == 'France') & (data['points'] >= 90) & (data['province'] == 'Alsace')]
mean = data.price.mean()
data['price_level'] = ['high' if i > mean else 'low' for i in data.price]
data.loc[:10,['price_level','price']]
print(data['country'].value_counts(dropna=False))
data.boxplot(column = 'points')
turkey_wine = data[(data['country'] == 'Turkey')]
morocco_wine = data[(data['country'] == 'Morocco')]
concatenating_data = pd.concat([turkey_wine,morocco_wine], axis = 0, ignore_index = True)
concatenating_data
data1 = data.loc[:,['country','province','points','price','price_level']]
data1
data1.dtypes
data1['province'] = data1['province'].astype('category')
data1['country'] = data1['country'].astype('category')
data1.info()
data1['price'].dropna(inplace = True)
assert data1['price'].notnull().all()
data1['price_level'].value_counts()
data2 = concatenating_data.set_index(['country','province'])
data2
data1.set_index(['country','province'])
mean_country = data1.groupby('country').mean()
mean_country
