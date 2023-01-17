import pandas as pd
import numpy as np

# Importing libraries for plotting the data:
import matplotlib.pyplot as plt
import seaborn as sns
malldata = pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
malldata.head()
malldata.isnull().sum()
malldata.drop(['CustomerID'], inplace=True, axis=1)
malldata.columns
import plotly.express as px

df = px.data.tips()

fig = px.histogram(malldata, x='Spending Score (1-100)', y = 'Annual Income (k$)',
                  color = 'Gender', hover_data = malldata.columns, marginal = 'box')

fig.show()

# Literally the first time I am using plotly and it is amazing:
import plotly.graph_objects as go
plt.figure(figsize=(10,10))
fig = go.Figure(data= [go.Mesh3d(x = malldata['Spending Score (1-100)'],
                                 y = malldata['Annual Income (k$)'],
                                 z = malldata['Age'],
                                 opacity = 0.5, color = 'skyblue')])

fig.update_layout(scene = dict(xaxis_title='Spending Score',
                               yaxis_title='Annual Income',
                               zaxis_title='Age'),
                  )

fig.show()
d = px.data.gapminder()
    
f = px.bar(malldata, x = malldata["Gender"], y = malldata["Spending Score (1-100)"],
           color = malldata["Gender"], animation_frame = malldata["Age"],
           range_y = [0,100])

fig.update_layout(scene = dict(xaxis_title='Gender',
                               yaxis_title='Spending Score (1-100)',
                               zaxis_title='Age'))
f.show()
sns.set_context("poster", font_scale=.7)
sns.set_palette(["skyblue", "magenta"])
plt.figure(figsize=(8,8))
sns.scatterplot(data=malldata, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender')
encdata = malldata.copy()

from sklearn.preprocessing import LabelEncoder

lab = LabelEncoder()

encdata['Gender'] = lab.fit_transform(encdata['Gender'])

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

scaleddata = scale.fit_transform(encdata)
from sklearn.cluster import KMeans

km = KMeans(n_clusters = 4, init="k-means++")

cluster = pd.DataFrame(km.fit_predict(scaleddata), columns=['cluster'])

newdata = pd.concat([cluster, malldata], axis=1)
plt.figure(figsize=(5,5))
sns.pairplot(newdata, hue='cluster')