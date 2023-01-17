import numpy as np 
import pandas as pd 
import seaborn as sns 
import scipy 
import sklearn 
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
import plotly.express as px
%matplotlib  inline
from sklearn.cluster import KMeans


df = pd.read_csv('../input/usa-cers-dataset/USA_cars_datasets.csv')

df.head(2)
df.describe() 
df.mean()
df.info ()
df.head()
X = df.drop('country', axis = 1 )
X.head()
X = X[[ 'price', 'brand',  'model', 'year', 'title_status', 'mileage','color','vin','lot','state', 'condition', ]]
X.head(20)
X.shape
X = X.fillna('')
X_transformed=X[X.year >=2006]
#sns.boxplot(X.year)
X_transformed.year.hist()
sns.boxplot(X_transformed.year)
X.year.unique()
X_transformed.year.unique()
X.title_status.unique()
X.brand.unique()
sns.pairplot(X_transformed[['price', 'brand','mileage','year']], hue = 'year')
price = X.groupby('brand')['price'].max().reset_index()
price  = price.sort_values(by="price")
price = price.tail(15)
fig = px.pie(price,
             values="price",
             names="brand",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.005, textinfo="percent+label")
fig.show()
year = px.data.iris()
fig = px.scatter(df, x="year", y="price", )
fig.show()
#df3 = pd.read_csv('USA_cars_datasets.csv')
#X[["F"]] = X3[["D", "E"]].astype(int)
#X['condition'] = X['condition'].astype(str)
X_3=X[X.year == 2006]
X_transformed= px.data.tips()
fig = px.sunburst(df, path=['year', 'title_status', 'brand'], values='price')
fig.show()
X.mean()
#kmeans = KMeans(n_clusters = 6, random state=1)f.fit(X.mean())
sns.heatmap(X.corr())
df['brand'].value_counts()[:30]
df['brand'].value_counts()[:30].plot(kind='barh')
    
X['state'].value_counts()[:50]
X['state'].value_counts()[:50].plot(kind='bar',)
X['title_status'].value_counts()[:2]
labels = 'clean vehicle', 'salvage insurance'
sizes = [2336,163]
explode = (0, 0.5) 

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=180)
ax1.axis('on')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
#percentage of total vehicles salvage/clean
X['year'].value_counts()[:30]
year = px.data.iris()
fig = px.scatter(df, x="year", y="brand")
fig.show()
X['year'].value_counts()[:30].plot(kind='bar',)
year = px.data.iris()
fig = px.scatter(X, x="year", y="state")
fig.show()
