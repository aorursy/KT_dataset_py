# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import datetime

import seaborn as sns

from dateutil.parser import parse

%matplotlib inline

import seaborn as sns
spotify = pd.read_csv('/kaggle/input/spotify1.csv',error_bad_lines=False)

spotify.head(10)
#it shows my music streaming history from 2018-10-16 to 2019-3-01

start_date = (spotify['end_time'].head())

start_date
end_date = (spotify['end_time'].tail())

end_date
#143 days passed

days_passed = (parse(spotify['end_time'].at[9992]) - parse(spotify['end_time'].at[1])).days

days_passed
# My spotify streaming peaked on 2018-10-22 at 339 songs.

top_dates_by_num_of_songs = spotify.groupby([spotify.end_time.str[:10]]).size().sort_values(ascending = False)

top_dates_by_num_of_songs
# My spotify streaming peaked on 2018-10-22 at 339 songs.

songs_by_day = spotify.groupby([spotify.end_time.str[:10]]).size().values

songs_by_day
date = spotify.groupby([spotify.end_time.str[:10]]).size().keys().values

date
# chart: daily listening trend in 6 monthts

plt.figure(figsize=(10,10))

fig, ax = plt.subplots()

ax.plot(date,songs_by_day)
#I streamed 9993 times on Spotify in 6 months

len(spotify)
#assign uuid to the artists

import uuid
for artist_name in spotify['artist_name'].unique():

    spotify.loc[spotify['artist_name'] == artist_name, 'artist_name_UUID'] = uuid.uuid4()
spotify.dtypes
#no null values

spotify.isnull().sum()
#I streamed 1,923 differente artists in 6 months

unique_artists=spotify.artist_name.unique()

len(unique_artists)
#my top artists!

top_artisits = spotify.artist_name.value_counts().head(10)

top_artisits
#my top songs!

spotify.track_name.value_counts().head(10)
#need to handle the timestamp

ms=spotify.ms_played.sum()
#I have streamed 334 hours in 6 months on spotify

hours=ms/(1000*60)/60

hours
# On average day, I spent 2.4 hours streaming on Spotify

avg_per_day = hours / days_passed

avg_per_day
#note that the line dateimeindex is important - converting 'end_time' into actual date_time

spotify['end_time'] = pd.DatetimeIndex(spotify['end_time'])

spotify['end_time_date'] = spotify['end_time'].dt.date

spotify['end_time_month'] = spotify['end_time'].dt.month

spotify['end_time_dow'] = spotify['end_time'].dt.dayofweek

spotify.head()

dua_lipa = spotify.loc[spotify.artist_name =='Dua Lipa'].sort_values(['ms_played'],ascending = False)

dua_lipa.head(10)
dua_lipa.track_name.value_counts()
dua_lipa.track_name.unique()

dua_songs = pd.DataFrame(dua_lipa)

dua_songs.head(10)
len(dua_songs)
#I streamed/shuffled 339 songs on 2018-10-22 and Jan was the most streamed month

end_time_month_spotify=spotify.end_time_month.value_counts().head(10)

end_time_month_spotify.plot.bar()
top_10_artists=spotify.artist_name.value_counts().head(10)

top_10_artists.plot.bar()
spotify['end_time_month'].value_counts().sort_index().plot.line()

#reviews['points'].value_counts().sort_index().plot.bar()