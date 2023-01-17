import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import re

from textblob import TextBlob

from wordcloud import WordCloud



import os

for dirname, _, filenames in os.walk('/kaggle/input/canada-covid19-tweets'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

def clean_tweet(tweet):

    """

    Removes mentions, hashtags, retweets, hyperlinks, and words that are less than 3 characters from tweets

    Input: tweet (string) you want cleaned

    Output: tweet (string) with mentions, hashtags, retweets, hyperlinks, and small words removed

    """

    tweet= re.sub('@[A-Za-z0-9]+', '', tweet) #remove mentions

    tweet= re.sub('#', '', tweet) #remove hashtags

    tweet= re.sub('RT[\s]+', '', tweet) #remove retweets

    tweet= re.sub('https?:\/\/\S+', '', tweet) #remove hyperlinks

    tweet= re.sub(r'\b\w{1,3}\b', '', tweet) #remove small words

    return tweet



def subj_tweet(tweet):

    """

    Returns subjectivity of a tweet (whether a tweet is fact- or opinion-based)

    Input: tweet (string) with all mentions, hashtags, retweets, hyperlinks, and small words removed

    Output: float between 0 and 1 (0 means tweet is  fact, 1 means tweet is very much an opinion)

    """

    return TextBlob(tweet).sentiment.subjectivity



def pol_tweet(tweet):

    """

    Returns polarity of a tweet (how negative or positive a tweet is)

    Input: tweet (string) with all mentions, hashtags, retweets, hyperlinks, and small words removed

    Output: float between -1 and 1 (-1 means tweet is very negative, 1 means tweet is very positive)

    """

    return TextBlob(tweet).sentiment.polarity



def pol_percent(polarity):

    """

    Returns classification label as to whether a tweet is positive or negative

    Input: polarity (float) of a string

    Output: string ("pos"/"neg"/"neutral") indicating whether the tweet associated with the polarity is positive, negative, or neutral 

    """

    if polarity <0:

        return "neg"

    elif polarity >0:

        return "pos"

    else:

        return "neutral"



def build_cloud(words):

    """

    Creates word cloud image from a body of text.

    Input: body of text (string)

    Output: None (displays word cloud)

    """

    word_cloud= WordCloud(width= 1000, height= 600, random_state= 444, max_font_size= 110, stopwords=['twitter','this','that','with','have','because','them','their','what','when']).generate(words)

    plt.figure()

    plt.imshow(word_cloud)

    plt.axis('off')

    return 
df= pd.read_csv('../input/canada-covid19-tweets/canada_tweets_v2.csv')

df.head()
#we only need a few columns to perform our sentiment analysis

#it also looks like each tweet has been retrieved twice so we'll get rid of duplicates too

df_small= df[['text','timestamp','city']].copy()

df_small.drop_duplicates(inplace= True)

#to make it easier to count the number of tweets published each week, add two more columns of month and day tweet was published

df_small['timestamp']= pd.to_datetime(df_small['timestamp'])



months= df_small['timestamp'].dt.month

days= df_small['timestamp'].dt.day

df_small.loc[:,'month published']= months

df_small.loc[:,'day published']= days



print('Tweets retrieved:', np.shape(df_small)[0])
#now we can clean, get the subjectivity and polarity, and assign a "pos"/"neg"/"neutral" label for each of the tweets

df_small['text']= df_small['text'].apply(clean_tweet)

df_small.loc[:,'subjectivity']= df_small['text'].apply(subj_tweet)

df_small.loc[:,'polarity']= df_small['text'].apply(pol_tweet)

df_small.loc[:,'pos/neg/neutral']= df_small['polarity'].apply(pol_percent)



#view head of final dataframe we're working with

df_small.head()
#count the number of tweets published each week

wk1= df_small[(df_small['month published'] == 3) & (df_small['day published'].between(1,7))]

wk2= df_small[(df_small['month published'] ==3) & (df_small['day published'].between(8,14))]

wk3= df_small[(df_small['month published'] ==3) & (df_small['day published'].between(15,21))]

wk4= df_small[(df_small['month published'] ==3) & (df_small['day published'].between(22,28))]

wk5= df_small[(df_small['month published'] ==3) & (df_small['day published'].between(29,31))]

wk5_2= df_small[(df_small['month published'] ==4) & (df_small['day published'].between(1,4))]

wk6= df_small[(df_small['month published'] ==4) & (df_small['day published'].between(5,11))]

wk7= df_small[(df_small['month published'] ==4) & (df_small['day published'].between(12,18))]

wk8= df_small[(df_small['month published'] ==4) & (df_small['day published'].between(19,25))]

wk9= df_small[(df_small['month published'] ==4) & (df_small['day published'].between(26,30))]

wk9_2= df_small[(df_small['month published'] ==5) & (df_small['day published'].between(1,2))]

tweet_counts= np.array([np.shape(wk1)[0], np.shape(wk2)[0], np.shape(wk3)[0], np.shape(wk4)[0], np.shape(wk5)[0]+np.shape(wk5_2)[0],np.shape(wk6)[0], np.shape(wk7)[0], np.shape(wk8)[0], np.shape(wk9)[0]+np.shape(wk9_2)[0]])



#visualize the tweet counts

x_axis= np.array(['M 1-7', 'M 8-14', 'M 15-21', 'M 22-28', 'M 29-A 4', 'A 5-11', 'A 12-18', 'A 19-25', 'A 26-M 2'])

plt.figure()

plt.xlabel('Week')

plt.ylabel('Number of Tweets')

plt.bar(x_axis, tweet_counts)

#create word cloud. omit frequently used conjunctions and the word "twitter"

words= ' '.join([tweet for tweet in df_small['text']])

build_cloud(words)
#this may take a moment to run...

N,d= np.shape(df_small)

plt.figure()

plt.grid()

plt.xlabel('Subjectivity')

plt.ylabel('Polarity')

for i in range(N): #subjectivity is column 6, polarity is column 5

    plt.scatter(df_small.iloc[i,5], df_small.iloc[i,6], color= 'blue')
all_but8= df_small.drop(df_small[(df_small['month published'] ==4) & (df_small['day published'].between(19,25))].index)

words= ' '.join([tweet for tweet in all_but8['text']])

build_cloud(words)