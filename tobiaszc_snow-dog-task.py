import pandas as pd

import numpy as np
import os

paths = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        paths.append(os.path.join(dirname, filename))

paths.sort()

paths
track_cols = ["track ID", "track title", "artist name"]

track_set = pd.read_table(paths[0], header=None, names=track_cols)
user_cols = ["user ID", "track ID", "listen time"]

user_set = pd.read_table(paths[1], header=None, names=user_cols)
track_set.head()
track_set.shape
user_set.head()
user_set.shape
track_set.isnull().sum()
track_set["artist name"].value_counts().head(15)
track_set.shape
(track_set["artist name"].str.find("Unknown") != -1).sum()
track_set[track_set["track title"] == "[Unknown]"]
track_set = track_set[track_set["artist name"].str.find("Unknown") == -1]
(track_set["track ID"].str.find("Unknown") != -1).sum()
user_set.isnull().sum()
((user_set["track ID"].str.find("Unknown") != -1) & (user_set["user ID"].str.find("Unknown") != -1)).sum()
top5_songs = user_set["track ID"].value_counts().iloc[0:5]

top5_songs
for i, song in enumerate(top5_songs.index):

    print(i+1, track_set[track_set["track ID"] == song].values[0][1:3])

    
top10_users = user_set["user ID"].value_counts().iloc[0:10]

for i, user_id in enumerate(top10_users.index):

    print(i+1, user_id)
# lets count how many times every song appear in user_set

song_counts = user_set["track ID"].value_counts()



# create a dataframe from value_counts()

song_counts = pd.DataFrame({"track ID":song_counts.index,"count":song_counts.values})

song_counts
# I will store all artist and informations about them in dictionary. Key is the artist name, value is a list: list[0] tells us how many users listened to artist's songs,

# list[1] is the artist's song list

# Created dictionary with following template : {artist_name: [number_of_songs_plays, [songs_titles]]}

artist_dict = {artist : [0, list] for artist in track_set["artist name"].value_counts().index.values}
for i, (artist, informations) in enumerate(artist_dict.items()):

    

    # first I collect all songs of current artist from track_set and store it in artist_dict

    # this information will be helpfull in the next task

    artist_dict[artist][1] = track_set[track_set["artist name"] == artist]["track ID"].tolist()

    

    # I need info how many times every song appears, i take it from song_counts, I store this value in list and make a sum of this values

    artist_dict[artist][0] = sum(song_counts[song_counts["track ID"].isin(artist_dict[artist][1])]["count"].tolist())

    

    # I wrote a blockade because counting all artists takes too long :(

    if i == 100:

        break



# I need to sort artist_dict by value

sorted_artists = {key: value for key, value in sorted(artist_dict.items(), reverse=True, key=lambda item: item[1][0])}



# Printing result

for i, (k, v) in enumerate(sorted_artists.items()):

    if i == 5: break

    print(i+1, k, v[0])
# I should take the Pink Floyd song list from the previous task but since I couldn't count all I will do this especially for Pink Floyd just in caset 

artist = "Pink Floyd"

artist_dict[artist][1] = track_set[track_set["artist name"] == artist]["track ID"].tolist()

artist_dict[artist][0] = sum(song_counts[song_counts["track ID"].isin(artist_dict[artist][1])]["count"].tolist())

    

songs_from_Pink_Floyd = artist_dict["Pink Floyd"]
# To find 3 most listened songs I will borrow line of code from task 6 

top_3 = song_counts[song_counts["track ID"].isin(artist_dict[artist][1])]["track ID"].iloc[0:3].values.tolist()

top_3
# lets search by users who listened even 1 of top3 songs of Pink Floyd

users_listened_to_top3 = user_set[user_set['track ID'].isin(top_3)]

users_listened_to_top3
users = user_set['user ID'].unique()
boolean_table=[]

for user in users:

    

    # I return true if user listened to all three songs, false otherwise

    boolean_table.append(len(users_listened_to_top3.loc[users_listened_to_top3['user ID'] == user, "track ID"].unique()) == 3)

    

# then return filtered list

users_listened_to_top_3 = users[boolean_table]
users_listened_to_top_3.size
import pandas as pd

import numpy as np

import sklearn 

import matplotlib.pyplot as plt
user_events = user_set[user_set['user ID'] == '0478c8abd9327b47848aa71c46112192'].sort_values('listen time')

user_events
user_events["shifted time"] = user_events["listen time"].shift(-1)

user_events["duration time"] = user_events["shifted time"] - user_events["listen time"]

user_events
user_events = user_events[user_events['duration time'] < 60*60]
user_events.drop(user_events.loc[:, ['listen time', 'shifted time']], axis=1, inplace=True)

user_events
All_I_Need_ID = track_set.loc[track_set['track title'] == 'All I Need', 'track ID'].values[0]

time_duration = user_events.loc[user_events["track ID"] == All_I_Need_ID, 'duration time']

All_I_Need_ID
n, bins, _ = plt.hist(time_duration, bins=20)
max_n = max(n)

max_n_index = np.where(n==max_n)[0][0]

duration_time = (bins[max_n_index] + bins[max_n_index+1]) / 2
from time import gmtime

from time import strftime

strftime("%H:%M:%S", gmtime(duration_time))
listen_sorted = user_set.copy()

listen_sorted.sort_values("listen time", inplace=True)

listen_sorted
samples = []

for i, us in enumerate(users):

    # I create condition table which will be userd to search and drop

    condition = listen_sorted['user ID'] == us

    

    # same as counting for one user

    u_events = listen_sorted.loc[condition]

    u_events["shifted time"] = u_events["listen time"].shift(-1)

    u_events["duration time"] = u_events["shifted time"] - u_events["listen time"]

    samples = np.concatenate((samples ,u_events.loc[(u_events["track ID"] == All_I_Need_ID) & (u_events['duration time'] < 60*60), 'duration time'].values))

    

    # I remove user from the searched dataset because I will no longer need him and it. This reduces the search set.

    listen_sorted = listen_sorted[~condition]

    if i == 5:

        break



print(samples)
n, bins = np.histogram(samples, bins = 200)

max_n = max(n)

max_n_index = np.where(n==max_n)[0][0]

duration_time = (bins[max_n_index] + bins[max_n_index+1]) / 2

strftime("%H:%M:%S", gmtime(duration_time))