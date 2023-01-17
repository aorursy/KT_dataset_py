# import data



import pandas as pd

import seaborn as sns

import matplotlib



listings=pd.read_csv('../input/winemag-data_first150k.csv',encoding="latin-1") 
# count how many data to start with



listings.count(axis=0)
listings.head(n=10)
# Remove 'region_2' column since too much na values exists

listings_new=listings.drop('region_2',axis=1)

listings_new=listings_new.dropna(axis=0,how='any')

# check how many rows are still left for exploration



listings_new.count(axis=0)
# Average price of wine produced in each country 



import numpy as np

import matplotlib.pyplot as plt

sns.boxplot(x='country',y='price',data=listings_new)

ax=plt.gca()

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')

ax.set_ybound(lower=None, upper=300)

plt.show()
# explorer relationship between wine ratings and price

sns.lmplot(x='points',y='price',data=listings_new)

plt.show()
average_price=pd.pivot_table(listings_new, values='price',index='country',aggfunc='mean').round(2)

average_price
# Average rating of wines from each country

sns.boxplot(x='country',y='points',data=listings_new)

ax=plt.gca()

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')

ax.set_ybound(lower=None, upper=None)

plt.show()
average_points=pd.pivot_table(listings_new, values='points',index='country',aggfunc='mean').round(0)

average_points
# Country having the largest number of wineries producing Chardonnay



Chardonnay=listings_new[listings_new['variety']=='Chardonnay']

Chardonnay_max=pd.pivot_table(Chardonnay,values='variety',index='country',aggfunc='count')

Chardonnay_max
# The chardonnay brand having the highest rating



Chardonnay_highrate=Chardonnay['points'].max()

Chardonnay[Chardonnay['points']==Chardonnay_highrate]
# Country having the largest number of wineries producing Pinot Noir



Pinor_Noir=listings_new[listings_new['variety']=='Pinot Noir']

Pinor_Noir_max=pd.pivot_table(Pinor_Noir,values='variety',index='country',aggfunc='count')

Pinor_Noir_max
# The Pinot Noir brand having the highest rating



Pinor_Noir_highrate=Pinor_Noir['points'].max()

Pinor_Noir[Pinor_Noir['points']==Pinor_Noir_highrate]
# Visualize the number of wineries in each state in US



listings_US=listings_new[listings_new['country']=='US']

states=pd.pivot_table(listings_US,values='designation',index='province',aggfunc='count')

states
import numpy as np



num_wineries=states.as_matrix(columns=None)

states_name=np.asarray(['AZ','CA','CO','CT','ID','IA','KT','MA',

                        'MI','MO','NV','NJ','NM','NY','NC','OH',

                        'OR','PA','TX','VT','VA','WA'])
import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode



pd.options.mode.chained_assignment = None

init_notebook_mode()



winery_scale=[[0, 'rgb(232, 213, 255)'], [100, 'rgb(218, 188, 255)']]



data = [dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = winery_scale,

        showscale = False,

        locations = states_name,

        locationmode = 'USA-states',

        z = num_wineries,

        marker = dict(

            line = dict(

                color = 'rgb(255, 255, 255)',

                width = 2)

            )

        )]



layout = dict(

         title = 'Wineries in US',

         geo = dict(

             scope = 'usa',

             projection = dict(type = 'albers usa'),

             countrycolor = 'rgb(255, 255, 255)',

             showlakes = True,

             lakecolor = 'rgb(255, 255, 255)')

         )



figure = dict(data = data, layout = layout)

iplot(figure)