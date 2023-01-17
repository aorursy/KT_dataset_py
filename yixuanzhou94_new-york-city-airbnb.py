import geopandas as gpd            # Python Geospatial Data Library

import plotly as plotly                # Interactive Graphing Library for Python

import plotly.express as px

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot, plot

init_notebook_mode(connected=True)



'''Spatial Visualizations'''

import folium

import folium.plugins



'''NLP - WordCloud'''

import wordcloud

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

'''Machine Learning'''

import sklearn

from sklearn import preprocessing

from sklearn import metrics

from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing necessery libraries for future analysis of the dataset

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline

import seaborn as sns
airbnb = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

airbnb.head()
#checking type of every column in the dataset

print(airbnb.dtypes)

#checking amount of rows in given dataset to understand the size we are working with

len(airbnb)
print('\nUnique values :  \n',airbnb.nunique())
airbnb.isnull().sum().sort_values(ascending=False)
total = airbnb.isnull().sum().sort_values(ascending=False)

percent = ((airbnb.isnull().sum())*100)/48895

missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'], sort=False).sort_values('Total', ascending=False)

missing_data.head(40)
total = airbnb.isnull().sum().sort_values(ascending=False)

percent = ((airbnb.isnull().sum())*100)/airbnb.isnull().count().sort_values(ascending=False)

percent = round(percent, 2)

missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'], sort=False).sort_values('Total', ascending=False)

missing_data.head(40)
#dropping columns that are not significant or could be unethical to use for our future data exploration and predictions

airbnb.drop(['id','host_name','last_review'], axis=1, inplace=True)

airbnb.head()
#replacing all NaN values in 'reviews_per_month' with 0

airbnb.fillna({'reviews_per_month':0}, inplace=True)

#airbnb['reviews_per_month'] = airbnb['reviews_per_month'].replace(np.nan, 0)

#examing changes

airbnb.reviews_per_month.isnull().sum()
airbnb.neighbourhood_group.unique()
airbnb.neighbourhood.unique()
airbnb.room_type.unique()
airbnb.corr().style.background_gradient(cmap='coolwarm')

#No strong correlation except number_of_reviews vs reviews_per_month
f,ax=plt.subplots(1,2,figsize=(18,8))

airbnb['neighbourhood_group'].value_counts().plot.pie(explode=[0,0.05,0,0,0],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Share of Neighborhood')

ax[0].set_ylabel('Neighborhood Share')

sns.countplot(x = 'neighbourhood_group',data = airbnb, ax=ax[1],order=airbnb['neighbourhood_group'].value_counts().index)

ax[1].set_title('Share of Neighborhood')

plt.show()
#what hosts (IDs) have the most listings on Airbnb platform and taking advantage of this service

top_host=airbnb.host_id.value_counts().head(10)

top_host
#coming back to our dataset we can confirm our fidnings with already existing column called 'calculated_host_listings_count'

top_host_check=airbnb.calculated_host_listings_count.max()

top_host_check
#setting figure size for future visualizations

sns.set(rc={'figure.figsize':(10,8)})

viz1 = top_host.plot(kind = 'bar')

viz1.set_title('Hosts with the most listings in NYC')

viz1.set_xlabel('Host IDs')

viz1.set_ylabel('Count of listings')

viz1.set_xticklabels(viz1.get_xticklabels(), rotation = 45)
#Brooklyn

sub_1=airbnb.loc[airbnb['neighbourhood_group'] == 'Brooklyn']

price_sub1=sub_1[['price']]

#Manhattan

sub_2=airbnb.loc[airbnb['neighbourhood_group'] == 'Manhattan']

price_sub2=sub_2[['price']]

#Queens

sub_3=airbnb.loc[airbnb['neighbourhood_group'] == 'Queens']

price_sub3=sub_3[['price']]

#Staten Island

sub_4=airbnb.loc[airbnb['neighbourhood_group'] == 'Staten Island']

price_sub4=sub_4[['price']]

#Bronx

sub_5=airbnb.loc[airbnb['neighbourhood_group'] == 'Bronx']

price_sub5=sub_5[['price']]

#putting all the prices' dfs in the list

price_list_by_n=[price_sub1, price_sub2, price_sub3, price_sub4, price_sub5]
#creating an empty list that we will append later with price distributions for each neighbourhood_group

price_list_by_n_2 = []

#creating list with known values in neighbourhood_group column

column_names = ['Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bronx']

#creating a for loop to get statistics for price ranges and append it to our empty list

for x in price_list_by_n:

    i = x.describe(percentiles=[.25, .50, .75])

    #i= i.iloc[3:]

    i.reset_index(inplace=True)

    i.rename(columns={'index':'Stats'}, inplace=True)

    price_list_by_n_2.append(i)

#changing names of the price column to the area name for easier reading of the table    

price_list_by_n_2[0].rename(columns={'price':column_names[0]}, inplace=True)

price_list_by_n_2[1].rename(columns={'price':column_names[1]}, inplace=True)

price_list_by_n_2[2].rename(columns={'price':column_names[2]}, inplace=True)

price_list_by_n_2[3].rename(columns={'price':column_names[3]}, inplace=True)

price_list_by_n_2[4].rename(columns={'price':column_names[4]}, inplace=True)

#finilizing our dataframe for final view    

stat_df = price_list_by_n_2

stat_df = [element.set_index('Stats') for element in stat_df]

stat_df = stat_df[0].join(stat_df[1:])

stat_df
#we can see from our statistical table that we have some extreme values, therefore we need to remove them 

#for the sake of a better visualization

sub_dataframe = airbnb[airbnb.price < 500]



viz2=sns.violinplot(data=sub_dataframe, x='neighbourhood_group', y='price')

viz2.set_title('Density and distribution of prices for each neighberhood_group')
#finding out top 10 neighbourhoods

airbnb.neighbourhood.value_counts().head(10)
#grabbing top 10 neighbourhoods for sub-dataframe

top_nei = airbnb.loc[airbnb['neighbourhood'].isin(['Williamsburg','Bedford-Stuyvesant','Harlem','Bushwick',

                 'Upper West Side','Hell\'s Kitchen','East Village','Upper East Side','Crown Heights','Midtown'])]



top_nei_plot = sns.catplot(x='neighbourhood', hue='neighbourhood_group', col='room_type', data=top_nei, kind='count')

top_nei_plot.set_xticklabels(rotation=90)
scatterplot = sub_dataframe.plot(kind='scatter', x='longitude', y='latitude', label='availability_365', c='price',

                  cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, figsize=(10,8))

scatterplot.legend()
import urllib

#initializing the figure size

plt.figure(figsize=(10,8))

#loading the png NYC image found on Google and saving to my local folder along with the project

i=urllib.request.urlopen('https://upload.wikimedia.org/wikipedia/commons/e/ec/Neighbourhoods_New_York_City_Map.PNG')

nyc_img=plt.imread(i)

#scaling the image based on the latitude and longitude max and mins for proper output

plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])

ax=plt.gca()

#using scatterplot again

sub_dataframe.plot(kind='scatter', x='longitude', y='latitude', label='availability_365', c='price', ax=ax, 

           cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, zorder=5)

plt.legend()

plt.show()
plt.figure(figsize=(10,6))

sns.scatterplot(airbnb.longitude,airbnb.latitude,hue=airbnb.neighbourhood_group)

plt.show()
import folium

from folium.plugins import HeatMap

m = folium.Map([40.7128,-74.0060],zoom_start=11)

HeatMap(airbnb[['latitude','longitude']].dropna(),radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)

display(m)
#initializing empty list where we are going to put our name strings

_names_=[]

#getting name strings from the column and appending it to the list

for name in airbnb.name:

    _names_.append(name)

#setting a function that will split those name strings into separate words   

def split_name(name):

    spl=str(name).split()

    return spl

#initializing empty list where we are going to have words counted

_names_for_count_=[]

#getting name string from our list and using split function, later appending to list above

for x in _names_:

    for word in split_name(x):

        word=word.lower()

        _names_for_count_.append(word)
from collections import Counter

#let's see top 25 used words by host to name their listing

top_25_w=Counter(_names_for_count_).most_common()

top_25_w=top_25_w[0:25]
#now let's put our findings in dataframe for further visualizations

top_25_w_df = pd.DataFrame(top_25_w)

top_25_w_df.head()
top_25_w_df.rename(columns={0:'Words', 1:'Count'}, inplace=True)

top_25_w_df.head()
#we are going to use barplot for this visualization

top_25_w_barplot = sns.barplot(x='Words', y='Count', data= top_25_w_df)

top_25_w_barplot.set_title('Counts of the top 25 used words for listing names')

top_25_w_barplot.set_ylabel('Count of words')

top_25_w_barplot.set_xlabel('Words')

top_25_w_barplot.set_xticklabels(top_25_w_barplot.get_xticklabels(), rotation=80)
top_reviewed_listings = airbnb.sort_values(by = ['number_of_reviews'], ascending = False)

top_reviewed_listings = top_reviewed_listings.head(10)

top_reviewed_listings
#let's grab 10 most reviewed listings in NYC

top_reviewed_listings=airbnb.nlargest(10,'number_of_reviews')

top_reviewed_listings
price_avg=top_reviewed_listings.price.mean()

print('Average price per night: {}'.format(price_avg))
#Categorising based on Price

def rank_price(hotel_price):

    if hotel_price <= 75:

        return 'Low'

    elif hotel_price >75 and hotel_price <= 500:

        return 'Medium'

    else:

        return 'High'
airbnb['price'].apply(rank_price).value_counts().plot(kind='bar');
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in airbnb.name)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="green").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
#Displaying rooms with maximum Reviews

roomwmaxreview = airbnb.sort_values(by=['number_of_reviews'],ascending=False).head(1000)

roomwmaxreview.head()
import folium

from folium.plugins import MarkerCluster

from folium import plugins

print('Rooms with the most number of reviews')



Long=-73.80

Lat=40.80

map_ = folium.Map([Lat,Long],zoom_start=10,)



map_rooms_map=plugins.MarkerCluster().add_to(map_)



for lat,lon,label in zip(roomwmaxreview.latitude,roomwmaxreview.longitude,roomwmaxreview.name):

    folium.Marker(location=[lat,lon],icon=folium.Icon(icon='home'),popup=label).add_to(map_rooms_map)



map_.add_child(map_rooms_map)



map_
airbnb.columns
airbnb.drop(['name'], axis = 1, inplace = True)

airbnb.head()
'''Encode labels with value between 0 and n_classes-1.'''

  

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()# Fit label encoder



airbnb['neighbourhood_group'] = le.fit_transform(airbnb['neighbourhood_group'])    

# Transform labels to normalized encoding.



le = LabelEncoder()



airbnb['neighbourhood'] = le.fit_transform(airbnb['neighbourhood'])



le = LabelEncoder()



airbnb['room_type'] = le.fit_transform(airbnb['room_type'])



airbnb.sort_values(by='price',ascending=True,inplace=True)



airbnb.head()
'''Train LRM'''

lm = LinearRegression()



X = airbnb.loc[:, airbnb.columns != 'price']

y = airbnb['price']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)



lm.fit(X_train,y_train)
'''Get Predictions & Print Metrics'''

predicts = lm.predict(X_test)



print("""

        Mean Squared Error: {}

        R2 Score: {}

        Mean Absolute Error: {}

     """.format(

        np.sqrt(metrics.mean_squared_error(y_test, predicts)),

        r2_score(y_test,predicts) * 100,

        mean_absolute_error(y_test,predicts)

        ))
error_airbnb = pd.DataFrame({

        'Actual Values': y_test,

        'Predicted Values': predicts}).head(20)



error_airbnb.head(5)
plt.figure(figsize=(16,8))

sns.regplot(predicts,y_test)

plt.xlabel('Predictions')

plt.ylabel('Actual')

plt.title("Linear Model Predictions")

plt.grid(False)

plt.show()