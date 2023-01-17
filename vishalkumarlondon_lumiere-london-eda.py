from IPython.display import Image
Image("../input/lumiereimages/header.png",
     width = 400)
Image("../input/lumiereimages/lumiere_map.png",
     width = 900)
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 300) # specifies number of rows to show
pd.options.display.float_format = '{:40,.0f}'.format # specifies default number format to 4 decimal places
pd.options.display.max_colwidth
pd.options.display.max_colwidth = 1000
installation_data = pd.read_csv('../input/lumierelondon/installation_data.csv')
installation_data.head()
installation_data = installation_data.fillna(value=0)
twitter_data = pd.read_csv('../input/lumierelondon/LumiereLDN_Twitter_clean.csv')
twitter_data.head(2)
twitter_data = twitter_data.fillna(value=0)

twitter_data['DateTime'] = pd.to_datetime(twitter_data['DateTime'])
twitter_data.dtypes
twitter_data['Hour'] = twitter_data['DateTime'].apply(lambda x: x.hour)
twitter_data['Month'] = twitter_data['DateTime'].apply(lambda x: x.month)
twitter_data['Day'] = twitter_data['DateTime'].apply(lambda x: x.day)
twitter_data['Year'] = twitter_data['DateTime'].apply(lambda x: x.year)
twitter_data.head(3)
twitter_data.columns
twitter = twitter_data[['DateTime', 'Hour', 'Day', 'Month', 'Year', 'UserId', 'UserHandle', 'Text',
       'UserFollowers', 'UserFriends', 'GeogCoordinates', 'Long', 'Lat',
       'UserLocation', 'UserLanguage', 'TweetURL','source', 'profile_image_url']]

twitter.head(3)
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go
from plotly.graph_objs import *

#You can also plot your graphs offline inside a Jupyter Notebook Environment. 
#First you need to initiate the Plotly Notebook mode as below:
init_notebook_mode(connected=True)
#which areas had the most instalations

installation_data['freq'] = installation_data.groupby('Location')['Location'].transform('count')

#make graph
data = [Bar(x=installation_data['Location'],
            y=installation_data['freq'])]

layout = Layout(
    title="Number of Lumiere Installations by Location",
    xaxis=dict(title='Location'),
    yaxis=dict(title='Number of Installations'),
    width = 500,
    
)

fig = Figure(data=data, layout=layout)

iplot(fig, filename='jupyter/basic_bar')
Image("../input/lumiereimages/installations1.png",
     width = 900)
Image("../input/lumiereimages/installations2.png",
     width = 900)
twitter.describe()
top10_twitterUsers = twitter[['UserHandle','UserFollowers','Text','DateTime']]
top10_twitterUsers = top10_twitterUsers.set_index(['DateTime'])

top10_twitterUsers = top10_twitterUsers.sort_values(['UserFollowers'],ascending=False).head(10)
top10_twitterUsers
print(top10_twitterUsers['UserHandle'])
twitter_data['DailyFreq'] = twitter_data.groupby('Day')['Day'].transform('count')

data = [Bar(x=twitter_data['Day'],  #change back to location_freq['Location']
            y=twitter_data['DailyFreq'])] #change back to location_freq['Frequency']

layout = Layout(
    title="Number of Tweets by Day",
    xaxis=dict(title='Day in January'),
    yaxis=dict(title='Number of Tweets'),
    width = 700
)

fig = Figure(data=data, layout=layout)

iplot(fig, filename='jupyter/basic_bar')
twitter_data['HourlyFreq'] = twitter_data.groupby('Hour')['Hour'].transform('count')

data = [Bar(x=twitter_data['Hour'],  #change back to location_freq['Location']
            y=twitter_data['HourlyFreq'])] #change back to location_freq['Frequency']

layout = Layout(
    title="Number of Tweets by Hour",
    xaxis=dict(title='Hour of Day'),
    yaxis=dict(title='Number of Tweets'),
    width = 700
)

fig = Figure(data=data, layout=layout)

iplot(fig, filename='jupyter/basic_bar')
twitter_data['LanguageFreq'] = twitter_data.groupby('UserLanguage')['UserLanguage'].transform('count')
df_lan = twitter_data[['UserLanguage','LanguageFreq']]
df_lan = df_lan.drop_duplicates()
df_lan = df_lan.sort_values('LanguageFreq', ascending=False)
df_lan = df_lan.reset_index(drop=True)
df_lan = df_lan.head(10)

df_lan
data = [
    go.Bar(
        x=df_lan['LanguageFreq'], # assign x as the dataframe column 'x'
        y=df_lan['UserLanguage'],
        orientation='h',
    )
]

layout = Layout(
    title="Frequency of Language by Twitter User",
    xaxis=dict(title='Number of Users'),
    yaxis=dict(title='User Language'),
    width = 700
)

fig = Figure(data=data, layout=layout)

iplot(fig)