# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from copy import deepcopy

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib
tweets_sanyam = pd.read_csv("/kaggle/input/ctds-interview-tweets/sanyam_episode_tweets.csv")

tweets_sanyam["twitter_reactions"] = tweets_sanyam['fav_count'] + tweets_sanyam['retweet_count']



follower_cnt = tweets_sanyam.sort_values(by='hero_follower_count',ascending=False)
## Code to plot the meta-data ##



def get_stacked_barcharts(ax, all_data, group_name, main_series_name, title, colours):

    '''

    Fancy function to plot stacked bar charts.

    Hopefully will be useful some day!

    '''

    sns.set_style("white")

    count = 0

    legend_dict = {}



    #Plot 1 - background - top series

    ax.bar(x=all_data[group_name] , height = all_data[main_series_name], color =colours[count], width=0.4)

    legend_dict[main_series_name] = matplotlib.patches.Rectangle((0,0),1,1,fc=colours[count])



    all_data.pop(main_series_name) #Removing the main series 

    

    #Plot 2 - overlay - bottom series

    for name, value in all_data.items():

        if name == group_name:

            continue

        count += 1

        ax.bar(x=all_data[group_name], height = value, color=colours[count], width=0.4)

        legend_dict[name] = matplotlib.patches.Rectangle((0,0),1,1,fc=colours[count])





    #Legend

    l = ax.legend(list(legend_dict.values()), list(legend_dict.keys()), loc=1, ncol = 1, prop={'size':10})

    l.draw_frame(False)



    #Titles

    sns.despine(left=True)

    ax.set_xlabel("Twitter Handler")

    ax.set_ylabel("Number Of Tweets")

    ax.title.set_text(title)

    



## Helper function for bar-charts ##

def get_graph(df, color_palette=["red", "darkblue"]):

    '''

    Helper function to plot bar charts

    '''

    sns.set_style("whitegrid")

    df_melt = pd.melt(df, id_vars = "heroes")

    fig = plt.figure()

    #For this particular notebook, x is always heroes

    chart = sns.barplot(x = 'heroes', y='value', hue = 'variable',data=df_melt, ci=None, \

                        palette=color_palette)

    xlabels = chart.get_xticklabels()

    chart.set_xticklabels(['']+xlabels, rotation=90, horizontalalignment='right');

    return chart;
## Metainfo in a text file. Has been copied for now. Have to automate to be read from a text file

## ---- Extracted on July 3rd ----



colours = ['aqua','powderblue','lightskyblue']

sns.set_context({"figure.figsize": (10, 10)})



tweet_metadata = {

                    'twitter_handler': ['sanyam_bhutani', 'ctds'],

                    'total_tweets': [2438, 48],

                    'inrange_tweets': [1334, 26]

                 }



inrange_metadata = {

                    'twitter_handler': ['sanyam_bhutani', 'ctds'],

                    'total_tweets': [1334, 26],

                    'interview_tweets': [283, 16],

                    'ama_tweets': [40, 2]

                   }



fig, (ax1,ax2) = plt.subplots(nrows=2)

fig.subplots_adjust(wspace=0.3, hspace=0.3)



title = "Total Tweets vs In-range tweets"

get_stacked_barcharts(ax1, deepcopy(tweet_metadata), 'twitter_handler', 'total_tweets', title, colours)

title = "Count of Normal, Interview-related, and AMA tweets within In-range tweets "

get_stacked_barcharts(ax2, deepcopy(inrange_metadata), 'twitter_handler', 'total_tweets',title, colours)

plt.show()

plt.close()
sns.color_palette("ch:2.5,-.2,dark=.3")

plt.figure(figsize=(10,5))

chart = sns.barplot( y=follower_cnt["hero_follower_count"],

                     x=follower_cnt['heroes'],

                     palette=("Blues_d"))

xlabels = chart.get_xticklabels()

chart.set_xticklabels(['']+xlabels, rotation=90, horizontalalignment='right');

plt.show()

plt.close()
tweets_sanyam[['hero_follower_count', 'episode_hero_tweet', 'tweet_id', 'created_date', 'full_text','user_mentions', 'retweet_count', 'fav_count']].head()
temp_df = tweets_sanyam[["heroes","youtube_likes","twitter_reactions"]]

temp_df = temp_df.sort_values(by=["youtube_likes"], ascending=[False])





chart = get_graph(temp_df) 

plt.show()
temp_df = tweets_sanyam[["heroes","youtube_watch_hours","twitter_reactions"]]

temp_df = temp_df.sort_values(by=["youtube_watch_hours"], ascending=[False])



chart = get_graph(temp_df) 

plt.show()
temp_df = tweets_sanyam[["heroes","youtube_watch_hours","youtube_views","hero_follower_count"]]

temp_df = temp_df[temp_df.columns.difference(['heroes'])].apply(np.log, axis=1)

temp_df["heroes"] = tweets_sanyam["heroes"]



temp_df = temp_df.sort_values(by=["hero_follower_count"], ascending=[False])



palette=["darkblue","orange", "green"]



chart = get_graph(temp_df,color_palette=palette) 

plt.show()
temp_df = tweets_sanyam[["heroes","spotify_streams","apple_listened_hours","twitter_reactions"]]

temp_df = temp_df.sort_values(by=["spotify_streams","apple_listened_hours"], ascending=[False,False])



palette=["orange", "green","darkblue"]

chart = get_graph(temp_df,color_palette=palette) 

plt.show()
tweets_sanyam["listeners"] = np.log(tweets_sanyam['anchor_plays'] + tweets_sanyam['spotify_listeners'] + tweets_sanyam['apple_listeners'])

tweets_sanyam["heroes_follower_count"] = np.log(tweets_sanyam['hero_follower_count'])

temp_df = tweets_sanyam[["heroes","heroes_follower_count","listeners"]]

temp_df = temp_df.sort_values(by=["heroes_follower_count"], ascending=[False])



palette=["darkblue","orange"]

chart = get_graph(temp_df, color_palette=palette) 

plt.show()
yt_anchors = tweets_sanyam[["heroes","anchor_plays","youtube_views"]]



yt_anchors = yt_anchors.sort_values(by=["youtube_views","anchor_plays"], ascending=[False,False])

palette=["orange", "red"]

chart = get_graph(yt_anchors,color_palette=palette) 

plt.show()