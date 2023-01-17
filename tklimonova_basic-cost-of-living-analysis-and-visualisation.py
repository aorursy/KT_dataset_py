#libraries

import numpy as np

import pandas as pd

import os

import math

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

%matplotlib inline

from plotly import tools

from wordcloud import WordCloud

import plotly_express as px    

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot
#DataFrame for the numbeo file

#dtypes = {'country': 'category', 'city': 'category'}

col_filepath = "../input/col-1718/cost_of_living_2017.csv"

col_data_2017 = pd.read_csv(col_filepath, index_col='rank')
#Cost of living in the world

avarage_col_per_country = col_data_2017.groupby('country')['col'].mean().reset_index()

plt.figure(dpi=150)

plt.title("Cost of living in the world in 2017", fontsize = 12)

sns.distplot(avarage_col_per_country['col'], hist=True, kde=False, color='green', bins=10)

plt.ylabel("Amount of countries", fontsize = 9) 

plt.xlabel("Cost of living", fontsize = 9) 

plt.xticks(np.arange(0, 150, 10), fontsize = 7)

plt.yticks(fontsize = 7)

plt.show()
#let's calculate top 10 countries with the highest cost of living

top_col_countries = col_data_2017.groupby('country')['col'].mean().reset_index().sort_values('col', ascending = False).head(10).set_index('country')
#Top 10 most expencive countries by cost of living in 2017

plt.figure(dpi=150)

plt.title("Top 10 most expencive countries by cost of living in 2017", fontsize = 12)

sns.barplot(top_col_countries.index, top_col_countries['col'], palette='deep')

plt.ylabel("Countries", fontsize = 9) 

plt.xlabel("Cost of Living", fontsize = 9)

plt.xticks(rotation = 45, fontsize = 7)

plt.yticks(fontsize = 7)

plt.show()
#let's calculate top 10 countries with the highest rent index in 2017

top_rent_countries = col_data_2017.groupby('country')['rent_ind'].mean().reset_index().sort_values('rent_ind', ascending = False).head(10).set_index('country')
#Top 10 countries with the highest rent index in 2017

plt.figure(dpi=150)

plt.title("Top 10 countries with the highest rent index in 2017", fontsize = 12)

sns.barplot(top_rent_countries['rent_ind'], top_rent_countries.index, palette='rocket')

plt.ylabel("Countries") 

plt.xlabel("Rent index") 

plt.show()
#let's calculate top 10 cities with the highest rent index in 2017

top_rent_cities = col_data_2017.sort_values('rent_ind', ascending = False).head(10).set_index('city')
#Top 10 cities with the highest rent index in 2017

plt.figure(dpi=150)

plt.title("Top 10 cities with the highest rent index in 2017", fontsize = 12)

sns.barplot(top_rent_cities['rent_ind'],top_rent_cities.index, palette='autumn')

plt.ylabel("Cities") 

plt.xlabel("Rent index") 

plt.show()
#Let's add date column to the cost of living 2017 data frame

col_data_2017['date']= 2017
#Cost of living vs rent index in 2017

fig = px.scatter(col_data_2017, x="col", y="rent_ind", color="country",

    size='purchas_pow_ind', hover_data=['city'],

    title='Rent index vs Cost of living in 2017'

)

fig.show()
#DataFrame for the Big Mac Index with memory reducing steps

cols=['date','country','dollar_price','country_code']

bmi_filepath = "../input/big-mac-index-with-date/big-mac-index.csv"

bmi_data = pd.read_csv(bmi_filepath, index_col = 'country_code', low_memory=False, usecols=cols)
#let's create a data frame for the latest date in the file

bmi_data_year_sort = bmi_data.sort_values('date', ascending = False)

bmi_data_2019 = bmi_data_year_sort.iloc[:37, :]
#Let's represent countries from our Big Mac dataset as a Word Cloud

x_2019 = bmi_data.country[bmi_data.date == '09/07/2019']

plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=400

                         ).generate(" ".join(x_2019))

plt.imshow(wordcloud,interpolation='gaussian')

plt.axis('off')

plt.savefig('graph.png')



plt.show()
#filtering top 10 countries in 2019 by average dollar price per country

top_bmi_countries_2019 = bmi_data_2019.groupby('country')['dollar_price'].mean().to_frame().reset_index().sort_values('dollar_price', ascending = False).head(10).set_index('country')
#top 10 countries with the highest big mac index

plt.figure(dpi=150)

plt.title("Top 10 countries with the highest big mac index 2019", fontsize = 10)

sns.barplot(top_bmi_countries_2019['dollar_price'],top_bmi_countries_2019.index, palette='mako')

sns.set(font_scale = 0.5)

plt.ylabel("Countries") 

plt.xlabel("Price for big mac in USD") 

plt.show()