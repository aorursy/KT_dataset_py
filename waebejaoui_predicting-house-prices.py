import numpy as np

import pandas as pd

pd.set_option('display.max_columns', None)

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools

init_notebook_mode(connected=True)  

from plotly.tools import FigureFactory as ff

import pycountry

import random

import squarify

from collections import Counter

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt # Matplotlib is a python 2D plotting library

%matplotlib inline 

# A magic command that tells matplotlib to render figures as static images in the Notebook.



import seaborn as sns # Seaborn is a visualization library based on matplotlib (attractive statistical graphics).

sns.set_style('whitegrid') # One of the five seaborn themes

import warnings

warnings.filterwarnings('ignore') # To ignore some of seaborn warning msg



from scipy import stats, linalg



import folium # for map visualization

from folium import plugins

from mpl_toolkits.mplot3d import Axes3D

import folium
house = pd.read_csv("../input/kc_house_data.csv")

house.shape
house.head()
house.columns.values
def random_colors(number_of_colors):

    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

                 for i in range(number_of_colors)]

    return color
def simple_graph(dataframe,type_of_graph, top = 0):

    data_frame = house[dataframe].value_counts()

    layout = go.Layout()

    

    if type_of_graph == 'barh':

        top_category = get_list(house[dataframe].dropna())

        if top !=None:

            data = [go.Bar(

                x=top_category[1].head(top),

                y=top_category[0].head(top),

                orientation = 'h',

                marker=dict(color=random_colors(10), line=dict(color='rgb(8,48,107)',width=1.5,)),

                opacity = 0.6

            )]

        else:

            data = [go.Bar(

            x=top_category[1],

            y=top_category[0],

            orientation = 'h',

            marker=dict(color=random_colors(10), line=dict(color='rgb(8,48,107)',width=1.5,)),

            opacity = 0.6

        )]



    elif type_of_graph == 'barv':

        top_category = get_list(house[dataframe].dropna())

        if top !=None:

            data = [go.Bar(

                x=top_category[0].head(top),

                y=top_category[1].head(top),

                marker=dict(color=random_colors(10), line=dict(color='rgb(8,48,107)',width=1.5,)),

                opacity = 0.6

        )]

        else:

            data = [go.Bar(

                x=top_category[0],

                y=top_category[1],

                marker=dict(color=random_colors(10), line=dict(color='rgb(8,48,107)',width=1.5,)),

                opacity = 0.6

            )]      



    elif type_of_graph == 'pie':

        data = [go.Pie(

            labels = data_frame.index,

            values = data_frame.values,

            marker = dict(colors = random_colors(20)),

            textfont = dict(size = 20)

        )]

    

    elif type_of_graph == 'pie_':

        data = [go.Pie(

            labels = data_frame.index,

            values = data_frame.values,

            marker = dict(colors = random_colors(20)),

            textfont = dict(size = 20)

        )]

        layout = go.Layout(legend=dict(orientation="h"), autosize=False,width=700,height=700)

        pass

    

    fig = go.Figure(data = data, layout = layout)

    py.iplot(fig)

    

def get_list(col_name):

    full_list = ";".join('col_name')

    each_word = full_list.split(";")

    each_word = Counter(each_word).most_common()

    return pd.DataFrame(each_word)
simple_graph('waterfront','pie',5)
simple_graph('grade','pie',5)
simple_graph('bedrooms','pie',5)
fig, ax = plt.subplots(figsize=(12,4))

sns.boxplot(x = 'price', data = house, orient = 'h', width = 0.8, 

                 fliersize = 3, showmeans=True, ax = ax)

plt.show()
fig, ax = plt.subplots(figsize=(12,4))

sns.boxplot(x = 'bathrooms', data = house, orient = 'h', width = 0.8, 

                 fliersize = 3, showmeans=True, ax = ax)

plt.show()
fig, ax = plt.subplots(figsize=(12,4))

sns.boxplot(x = 'floors', data = house, orient = 'h', width = 0.8, 

                 fliersize = 3, showmeans=True, ax = ax)

plt.show()
sns.jointplot(x="sqft_living", y="price", data=house, kind = 'reg', size = 7)

plt.show()
sns.jointplot(x="bedrooms", y="price", data=house, kind = 'reg', size = 7)

plt.show()
sns.jointplot(x="bathrooms", y="price", data=house, kind = 'reg', size = 7)

plt.show()
f, axes = plt.subplots(1, 1,figsize=(15,5))

sns.boxplot(x=house['grade'],y=house['price'])

sns.despine(left=True, bottom=True)

axes.set(xlabel='grade', ylabel='Price')

axes.yaxis.tick_left()
f, axes = plt.subplots(1, 1,figsize=(15,5))

sns.boxplot(x=house['floors'],y=house['price'])

sns.despine(left=True, bottom=True)

axes.set(xlabel='floors', ylabel='Price')

axes.yaxis.tick_left()
f, axes = plt.subplots(1, 1,figsize=(15,5))

sns.boxplot(x=house['waterfront'],y=house['price'])

sns.despine(left=True, bottom=True)

axes.set(xlabel='waterfront', ylabel='Price')

axes.yaxis.tick_left()
f, axes = plt.subplots(1, 1,figsize=(15,5))

sns.boxplot(x=house['condition'],y=house['price'])

sns.despine(left=True, bottom=True)

axes.set(xlabel='condition', ylabel='Price')

axes.yaxis.tick_left()
fig=plt.figure(figsize=(19,12.5))

ax=fig.add_subplot(2,2,1, projection="3d")

ax.scatter(house['floors'],house['bedrooms'],house['bathrooms'],c="darkgreen",alpha=.5)

ax.set(xlabel='\nFloors',ylabel='\nBedrooms',zlabel='\nBathrooms')

ax.set(ylim=[0,12])



fig=plt.figure(figsize=(19,12.5))

ax=fig.add_subplot(2,2,1, projection="3d")

ax.scatter(house['waterfront'],house['bedrooms'],house['bathrooms'],c="darkgreen",alpha=.5)

ax.set(xlabel='\nwaterfront',ylabel='\nBedrooms',zlabel='\nBathrooms')

ax.set(ylim=[0,12])

fig=plt.figure(figsize=(19,12.5))

ax=fig.add_subplot(2,2,1, projection="3d")

ax.scatter(house['waterfront'],house['bedrooms'],house['sqft_living'],c="darkgreen",alpha=.5)

ax.set(xlabel='\nWaterfront',ylabel='\nBedrooms',zlabel='\nsqft_living')

ax.set(ylim=[0,12])
features = ['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront',

            'view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated',

            'zipcode','lat','long','sqft_living15','sqft_lot15']



mask = np.zeros_like(house[features].corr(), dtype=np.bool) 

mask[np.triu_indices_from(mask)] = True 



f, ax = plt.subplots(figsize=(16, 12))

plt.title('Pearson Correlation Matrix',fontsize=25)



sns.heatmap(house[features].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", #"BuGn_r" to reverse 

            linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});