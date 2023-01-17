import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input/covid19-canada-tweet-summary'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# load tweet summary data and inspect to understand what we're working with

df= pd.read_csv('../input/covid19-canada-tweet-summary/covid19_tweet_summary.csv')

df.head()
group_terms= df.groupby('Search terms')['Number of tweets'].sum()

print('Total tweets retrieved across Canada from March 1 to April 18, 2020:',np.sum(group_terms))

print('Search terms used:', df['Search terms'].unique())

print('Where tweets are from:', df['City +200km radius'].unique())
ax= group_terms.plot.bar(title= 'Count of Tweets Containing Certain Words, all cities')

ax.set_xlabel('Words Tweet Contains')

ax.set_ylabel('Number of Tweets')



plt.figure()

group_city= df.groupby('City +200km radius')['Number of tweets'].sum()

ax2= group_city.plot.bar(title= 'Count of Tweets from Each City, all search terms')

ax2.set_xlabel('City +200km radius')

ax2.set_ylabel('Number of Tweets')



plt.figure()

term_city= df.groupby(['City +200km radius', 'Search terms'])['Number of tweets'].sum()

ax3= term_city.plot.bar(title= 'Count of Tweets from Each City by Search Term')

ax3.set_xlabel('City +200km radius, Search terms')

ax3.set_ylabel('Number of tweets')
#get sum of each column of users tweeting and put in array so it can be easily plotted

users= np.array([['General public','Doctors/Healthcare organizations','News','Government'],[df["Who's tweeting - public"].sum(),df["Who's tweeting - doctors/healthcare org"].sum(),df["Who's tweeting - news"].sum(),df["Who's tweeting - gov"].sum()]])

print('Number of unique Twitter users tweeting about Covid-19 and cancer:', df['Number of unique users'].sum())

plt.pie(users[1,:], labels= users[0,:], rotatelabels= True)

plt.title("Distribution of Who's Tweeting about Covid-19 and Cancer")
group_date= df.groupby('Week number')['Number of tweets'].sum()

ax_date= group_date.plot.bar(title= 'Count of Tweets from Each Week (Mar 1 - Apr 18, 2020)')

ax_date.set_ylabel('Number of tweets')
#get sum of each column of counts of tweet content and put in array so it can be easily plotted

tweet_cont= np.array([['Virus concern','Cancer concern','Change in cancer treatment','Support/Info','Other'],[df['Content - virus fear/concern'].sum(), df['Content - cancer treatment concern'].sum(), df['Content - change to personal cancer treatment'].sum(), df['Content - support/information'].sum(),  df['Content - misc (not directly related to virus or cancer)'].sum()]])

plt.pie(tweet_cont[1,:], labels= tweet_cont[0,:], rotatelabels= True)