#credit to Martin Beck. Code samples taken from: 
#https://towardsdatascience.com/how-to-scrape-tweets-from-twitter-59287e20f0f1
!pip install GetOldTweets3

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import GetOldTweets3 as got

#import functions for part-of-speech tagging
from nltk import pos_tag, pos_tag_sents
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
#input data files are available in the "../input/" directory.
#for example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#any results you write to the current directory are saved as output.
#function the pulls tweets from a specific username and turns to csv file

#parameters: (list of twitter usernames), (max number of most recent tweets to pull from)
def username_tweets_to_csv(username, count):
    #creation of query object
    tweetCriteria = got.manager.TweetCriteria().setUsername(username)\
                                            .setMaxTweets(count)
    #creation of list that contains all tweets
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)

    #creating list of chosen tweet data
    user_tweets = [[tweet.date, tweet.text] for tweet in tweets]

    #creation of dataframe from tweets list
    tweets_df = pd.DataFrame(user_tweets, columns = ['Datetime', 'Text'])

    #converting dataframe to CSV
    tweets_df.to_csv('{}-{}k-tweets.csv'.format(username, int(count/1000)), sep=',')

#function that pulls tweets based on a general search query and turns to csv file

#parameters: (text query you want to search), (max number of most recent tweets to pull from)
def text_query_to_csv(text_query, count):
    #creation of query object
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(text_query)\
                                                .setMaxTweets(count)
    #creation of list that contains all tweets
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)

    #creating list of chosen tweet data
    text_tweets = [[tweet.date, tweet.text] for tweet in tweets]

    #creation of dataframe from tweets
    tweets_df = pd.DataFrame(text_tweets, columns = ['Datetime', 'Text'])

    #converting tweets dataframe to csv file
    tweets_df.to_csv('{}-{}k-tweets.csv'.format(text_query, int(count/1000)), sep=',')


#execution of this code block may take several minutes
#input Twitter username(s) to scrape tweets and name csv file
#change the count variable to change number of most recent Tweets to be scraped
username = 'sebastmarsh'
count = 3000

#calling function to turn username's past x amount of tweets into a CSV file
username_tweets_to_csv(username, count)

#place tweets into dataframe, drop columns, convert column type to string, tokenize.
df = pd.read_csv('sebastmarsh-3k-tweets.csv')
df = df.drop(columns=['Unnamed: 0', 'Datetime'])
df['Text'] = df['Text'].astype(str)
text = pos_tag_sents(df['Text'].apply(word_tokenize).tolist())
#prints the 2000th most recent Tweet
text[2000]
#remove all words not considered proper nouns (NNPs) in NLTK libray,
#counts number of occurrences for each
wordFrequency = list()
for line in text:
    for tag in line:
        if tag[1] == 'NNP':
            if tag[0].lower() not in wordFrequency:
                wordFrequency.append((tag[0].lower(),1))
            else:
                index = 0
                for word in wordFrequency:
                    if tag[0].lower() == word[index][0]:
                        word[index][1] = word[index][1] + 1
                    index = index + 1
#make downloadable as csv
df = pd.DataFrame(wordFrequency)
df.to_csv('wordFrequency.csv')