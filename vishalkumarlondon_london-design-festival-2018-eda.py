import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 300) # specifies number of rows to show
pd.options.display.float_format = '{:40,.0f}'.format # specifies default number format to 4 decimal places
pd.options.display.max_colwidth
pd.options.display.max_colwidth = 1000
# This line tells the notebook to show plots inside of the notebook
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sb
twitter_data = pd.read_excel('../input/ldf18_twitter.xlsx')

twitter_data.head(2)
twitter_data = twitter_data.fillna(value=0)
twitter_data['DateTime'] = pd.to_datetime(twitter_data['time'])
twitter_data.dtypes
twitter_data['Hour'] = twitter_data['DateTime'].apply(lambda x: x.hour)
twitter_data['Month'] = twitter_data['DateTime'].apply(lambda x: x.month)
twitter_data['Day'] = twitter_data['DateTime'].apply(lambda x: x.day)
twitter_data['Year'] = twitter_data['DateTime'].apply(lambda x: x.year)
twitter_data.head(3)
twitter_data.columns
twitter_data.columns = ['UserId', 'UserHandle', 'Text', 'created_at', 'time', 'GeogCoordinates',
       'UserLanguage', 'in_reply_to_user_id_str', 'in_reply_to_screen_name',
       'from_user_id_str', 'in_reply_to_status_id_str', 'source',
       'profile_image_url', 'UserFollowers', 'UserFriends',
       'UserLocation', 'TweetURL', 'entities_str', 'DateTime', 'Hour',
       'Month', 'Day', 'Year']

twitter = twitter_data[['DateTime', 'Hour', 'Day', 'Month', 'Year', 'UserId', 'UserHandle', 'Text',
       'UserFollowers', 'UserFriends', 'GeogCoordinates',
       'UserLocation', 'UserLanguage', 'TweetURL']]

twitter.head(3)
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go
from plotly.graph_objs import *

#You can also plot your graphs offline inside a Jupyter Notebook Environment. 
#First you need to initiate the Plotly Notebook mode as below:
init_notebook_mode(connected=True)
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
    xaxis=dict(title='Day in September'),
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
