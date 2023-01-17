import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from matplotlib import cm

sns.set_style('ticks')

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

plt.xkcd() 
Zo= pd.read_csv("../input/zomato.csv")

Zo.head()
Zo.drop(columns = ["url", "address","phone","location", "reviews_list", "menu_item", "listed_in(type)"], axis =1)
Zo['name'] = Zo['name'].str.strip()

Zo['listed_in(city'] = Zo['listed_in(city)'].str.strip()

Zo['cuisines'] = Zo['cuisines'].str.strip()

Zo['listed_in(type)'] = Zo['listed_in(type)'].str.strip()
sns.countplot(x=Zo['book_table'])

fig=plt.gcf()

fig.set_size_inches(6,4)
corr_mat = Zo.corr(method='pearson')

plt.figure(figsize=(20,10))

sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')
plt.matshow(Zo.corr())

plt.colorbar()

plt.show()
Zo['listed_in(city)'].value_counts().plot(kind='bar', title='Restaurant count by Area',figsize=(20,8)) 
plt.figure(figsize=(10,7))

sns.scatterplot(x="rate",y='approx_cost(for two people)',data=Zo)

plt.show()