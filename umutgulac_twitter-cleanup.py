# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import tweepy
import csv
import json
from datetime import date 
import time
import os
import sys
import codecs
import pandas as pd
from pandas.io.json import json_normalize
import pyquery

#Historical Tweets
from GetOldTweets import got3 as got

# load Twitter API credentials

with open('twitter_credentials.json') as cred_data: 
    info = json.load(cred_data)
    consumer_key = info['CONSUMER_KEY']
    consumer_secret = info['CONSUMER_SECRET']
    access_key = info['ACCESS_KEY']
    access_secret = info['ACCESS_SECRET']
    
def printExTweepy(status):
    if hasattr(status, "retweeted_status"):  # Check if Retweet
        try:
            print(status.retweeted_status.extended_tweet["full_text"])
        except AttributeError:
            print(status.retweeted_status.full_text)
    else:
        try:
            print(status.extended_tweet["full_text"])
        except AttributeError:
            print(status.full_text)

def cleanAllTweets():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    
####################### PARAMETRELER  #######################  

    max_tweets = int(5)
    start = time.time()

    # User to cleanup
    
    userName = '@babazmir'
    sinceDate = "2000-01-01"
    untilDate = "2020-12-31"
    #print("Twitter'dan " + userName + " kullanıcısına ait " + sinceDate + " ve " + untilDate + " tarihleri arasında atılan ... kadar tweet silinmiştir." + '\n' )
    
####################### PARAMETRELER  #######################

    def printTweet(descr, t):
        print(descr)
        print("Tweet ID: %s" % t.id)
        print("Tweet Date: %s" % t.formatted_date) 
        print("Username: %s" % t.username)
        print("Retweets: %d" % t.retweets)
        print("Text: %s" % t.text)
        print("Mentions: %s" % t.mentions)
        print("Hashtags: %s\n" % t.hashtags)
        
    # Get historical tweets from https://github.com/Jefferson-Henrique/GetOldTweets-python
    #tweetCriteria = got.manager.TweetCriteria().setQuerySearch('alibabacan').setSince("2019-12-01").setUntil("2019-12-24").setMaxTweets(max_tweets)
    #tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]
    #print("### Example 2 - Get tweets by query search [alibabacan] : " + str(type(tweet)) )
    
    print('User name is ' + userName + '\n')
    total_delete_count = 0
    
    #for i in range(len(tweets['text'])):
    #    if tweets['tweetos'].str.contains('@')[i]  == False:
    #        tweets['tweetos'][i] = 'other'
    
    for i in range(2):
        tweetCriteria = got.manager.TweetCriteria().setUsername(userName).setMaxTweets(max_tweets) #.setSince(sinceDate).setUntil(untilDate).
        searched_tweets = got.manager.TweetManager.getTweets(tweetCriteria)
        #alltweets.extend(searched_tweets)
        print('GetOldTweets ile ' + userName + ' kullanıcısına ait indirilen toplam tweet sayısı: ' + str(len(searched_tweets)) )
        print('GetOldTweets ile tweetlerden biri : ' + searched_tweets[0].id + ' & '   + searched_tweets[0].text   + ' gibi birşey olmalı' + '\n')

        delete_count = 0
        for status in searched_tweets: #tweepy.Cursor(api.user_timeline, user_id="@babazmir").items(max_tweets):
        # process status here
            print( status.id + '/' + status.text )
            delete_count += 1
            total_delete_count += 1
            #api.destroy_status(status.id)
            #API.favorites([id][, page])
            #api.destroy_favorite(id)

        print (userName + ' kullanıcısına ait ' + str(delete_count) + ' tweet silindi.' + '\n'+ '\n'  )
    
    
    
    print (userName + ' kullanıcısına ait TOPLAM ' + str(total_delete_count) + ' tweet silindi.' + '\n'+ '\n'  )
    
    #to_delete_ids = []
    # delete marked tweets by status ID
    #for status_id in to_delete_ids:
    #try:
    #    api.destroy_status(status_id)
    #    print(status_id, 'deleted!')
    #    delete_count += 1
    #except:
    #    print(status_id, 'could not be deleted.')
    #print(delete_count, 'tweets deleted.')
    
    end = time.time()
    print( "Elapsed time %s" % time.strftime("%H:%M:%S", time.gmtime(end-start)) )
    
    pass

if __name__ == '__main__':
    cleanAllTweets()
