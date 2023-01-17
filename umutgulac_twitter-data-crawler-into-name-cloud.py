# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

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

#Visualisation
import matplotlib.pyplot as plt
import matplotlib
import warnings
matplotlib.style.use('ggplot')
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

#NLP
from wordcloud import WordCloud, STOPWORDS # https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

#Historical Tweets
from GetOldTweets import got3 as got

# load Twitter API credentials

with open('twitter_credentials.json') as cred_data: 
    info = json.load(cred_data)
    consumer_key = info['CONSUMER_KEY']
    consumer_secret = info['CONSUMER_SECRET']
    access_key = info['ACCESS_KEY']
    access_secret = info['ACCESS_SECRET']
    
def printTweepy(status):
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
            
def wordcloud(tweets, col):  # wordcloud(tweets,col)
    stop_words = stopwords.words('turkish')
    ex_stop_words = ['alibabacan','yeni parti','yeni parti gerek','yeni','parti','gerek','ali','babacan']
    stop_words.extend(ex_stop_words)
    print("Konuyla ilgisiz olarak analizden çıkarılan kelimeler (STOPWORDS): " + str(ex_stop_words) ) 
    #wordcloud = WordCloud(background_color="white",stopwords=stop_words,random_state=2016).generate(" ".join([i for i in tweets]))  # generate(" ".join([i for i in tweets[col]]))
    wordcloud = WordCloud(background_color="white",stopwords=stop_words,max_words=20000,min_font_size=6,max_font_size=30,collocations=False).generate(" ".join([i for i in tweets[col]]))
    plt.figure( figsize=(20,10), facecolor='k')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Turkiyenin Politik Isim Haritasi")

def get_all_tweets():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    
####################### PARAMETRELER  #######################  

    max_tweets = int(50)
    start = time.time()

    # Mention the hashtag that you want to look out for
    
    hashtag = 'alibabacan_yeniparti_konulu'
    hashtagList = ['alibabacan','yeni parti','yeni parti gerek']
    sinceDate = "2020-05-01" # "2019-07-08"
    untilDate = "2020-05-31"
    # popular irrelevant tweets to be removed
    removeTweets = ["#EYT", "#SiziNedenSeçelim","#YıldızSuntaEmekçileriYalnızDeğildir"]
    # #alibabacan #yeniparti #onceinsan # alibabacanhaberturkte #babacanhaberturkte #dortegilim #yenipartigerekli #yeniolusum #yenipartielzemdir
    tweetsDir = 'D:\\Jupyter\\'
    print("Twitter'da " + hashtag + " olarak " + sinceDate + " ve " + untilDate + " tarihleri arasında atılan tweet ler incelenmiştir." + '\n' )
    
####################### PARAMETRELER  #######################

    fName = hashtag + '.txt'
    jsonfName = tweetsDir + 'tweets_about_' + hashtag + '.json'
    csvfName = tweetsDir + 'tweets_about_' + hashtag + '.csv'
    csvhashtagfName = tweetsDir + 'hashtagss_about_' + hashtag + '.csv'
    if os.path.exists(fName):
        os.remove(fName)
    if os.path.exists(jsonfName):
        os.remove(jsonfName)
    #if os.path.exists(csvfName):
        #os.remove(csvfName)
    
    # Print anything
    #print('\n'.join(sys.path)) 
    
    def printTweet(descr, t):
        print(descr)
        print("Tweet Date: %s" % t.formatted_date) 
        print("Username: %s" % t.username)
        print("Retweets: %d" % t.retweets)
        print("Text: %s" % t.text)
        print("Mentions: %s" % t.mentions)
        print("Hashtags: %s\n" % t.hashtags)
        
    def serialize(obj):
        if isinstance(obj, date):
            serial = obj.isoformat()
            return serial
        return obj.__dict__
    
    def serializeList(l):
        for i in range(len(l)):
            if isinstance(l[i], date):
                l[i] = l[i].isoformat()
        return l.__dict__
    
    # Get historical tweets from https://github.com/Jefferson-Henrique/GetOldTweets-python
    #tweetCriteria = got.manager.TweetCriteria().setQuerySearch('alibabacan').setSince("2019-12-01").setUntil("2019-12-24").setMaxTweets(max_tweets)
    #tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]
    #print("### Example 2 - Get tweets by query search [alibabacan] : " + str(type(tweet)) )
    
    alltweets = []
    i = 0
    while i < len(hashtagList):
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(hashtagList[i]).setSince(sinceDate).setUntil(untilDate).setMaxTweets(max_tweets)
        searched_tweets = got.manager.TweetManager.getTweets(tweetCriteria)
        print (hashtagList[i] + ' kelimesiyle ilgili ' + str(len(searched_tweets)) + ' tweet çekilmiştir.' )
        alltweets.extend(searched_tweets)
        i += 1
    
    print("İndirilen toplam tweet sayısı: " + str(len(alltweets)))
    
    #for tweet in alltweets:
    #    printTweet('\n' + "### TWEET", tweet)
    
    deduptweets = []
    for tweet in alltweets:
        deduptweets.append( serialize(tweet) )
        
    dedupnormtweets = json_normalize(deduptweets)
    dedupnormtweets.to_csv(csvfName, columns=['id','permalink','username','text','date','formatted_date','retweets','favorites','mentions','hashtags','geo','urls','author_id'], encoding='utf-8-sig', index=False, float_format='%.3f') #quotechar="\"",)
           
    jtweets = []
    for tweet in alltweets:
        jtweets.append( serialize(tweet) )
        if tweet.retweets > 0:
            for i in range(tweet.retweets):
                tweet.retweets = -1
                alltweets.append(tweet)
    
    print("Retweet lerle beraber toplam tweet sayısı: " + str(len(jtweets)))
    print("İlk tweet tarihi: " + jtweets[len(jtweets)-1]['formatted_date'] )
    print("Son tweet tarihi: " + jtweets[0]['formatted_date'] )
    
    #for tweet in alltweets:
    #    printTweet('\n' + "### TWEET", tweet)
        
    #print('\n' + '\n' + "JSON  : " + str(jtweets) )
    
    
    #######alltweets = []
    #######i = 0
    #######while i < len(hashtagList):
    #######    searched_tweets = [status._json for status in tweepy.Cursor(api.search, q=hashtagList[i], count=max_tweets, tweet_mode='extended', 
    #######        include_entities=True, wait_on_rate_limit=True).items(max_tweets)]
    #######    print ('Extracted ' + str(len(searched_tweets)) + ' tweets with ' + hashtagList[i])
    #######    #print( 'eski text : ' + searched_tweets[5]['full_text'] )
    #######    for j, element in enumerate(searched_tweets):
    #######        if 'retweeted_status' in searched_tweets[j]:
    #######            searched_tweets[j]['full_text'] = searched_tweets[j]['full_text'].partition(searched_tweets[j]['retweeted_status']['full_text'][0:9])[0] + searched_tweets[j]['retweeted_status']['full_text']
    #######            #print( 'yeni fulltext : ' + searched_tweets[5]['full_text'] )
    #######    alltweets.extend(searched_tweets)
    #######    i += 1
    # ,since="2019-12-18",until="2019-12-19"
      
    ######## remove Bir Damat tweet: 1206993123223707649
    #######for element in alltweets[:]:
    #######    if 'retweeted_status' in element:
    #######        if element['retweeted_status']['id'] == 1206993123223707649 :
    #######            #print( '\n' + 'Bir Damat tweet Deleted : '  + str( element['id']) )
    #######            alltweets.remove(element)
    #######            
    ######## remove popular irrelevant tweets
    #######for element in alltweets[:]:
    #######    if any(word in element['full_text'] for word in removeTweets):
    #######        #print( '\n' + 'ETY Tweet Deleted : '  + str( element['id']) )
    #######        alltweets.remove(element)
    #######        
    #######print( '\n' + 'After cleansing # of Tweets left : '  + str( len(alltweets) ) + '\n' )
                
    # check if removed
    #j = 0
    #for j, element in enumerate(alltweets):
    #    if 'retweeted_status' in element:
    #        if element['retweeted_status']['id'] == 1206993123223707649:
                #print('\n' + '\n' + 'Hala var aq : ' +'\n' + str(element['id']) +'\n' +  str(element['retweeted_status']['id']) + '\n' + str(element['retweeted_status']['full_text']   ) )
    
    #print(str(alltweets))
    
    #tweet_pyobj = json.loads(searched_tweets)
    tweets = json_normalize(jtweets)
    
    #print('\n' + '\n' + 'SEARCHED TWEETS: ' + '\n' + str( alltweets ) + '\n' +'\n')
    #print( 'NORMALIZED TWEETS: ' + '\n' + str( tweets ) )
    #print( '\n' +'\n' + 'NORMALIZED TWEETS: ' +  str(tweets['created_at'][10]) +'\n'+ str(tweets['id'][10]) +'\n' + tweets['full_text'][10] + '\n'+ '\n' )
    #print( '\n' +'\n' + tweets['created_at'][1] + '\n' + tweets['full_text'][1] + '\n' )
    #'retweeted_status': {'created_at'
     
        #for key in json.loads(json_strings[0]):
     #   print(key)
    #print( type(searched_tweets.id) )
    
    
    ######tweetags = [tweet.entities.get('hashtags') for tweet in tweepy.Cursor(api.search, q='alibabacan', count=max_tweets, tweet_mode='extended', 
    ######        include_entities=True, wait_on_rate_limit=True).items(max_tweets)]
    #######print('\n' + "tweetags : " + str(tweetags) )
    ######
    ######hashtags = []
    ######for i in range(len(tweetags)):
    ######    hashtags.append( [ sub['text'] for sub in tweetags[i] ] )
    ######    
    ######hTags = []
    ######for sublist in hashtags:
    ######    for item in sublist:
    ######        hTags.append(item)
    ######
    ######print( '\n' + '# of hastags  : '  + str( len(hTags) ) + '\n' )
    
    #print( '\n' + "tags : " + str(hTags) )
    
    #Preprocessing del RT @blablabla:
    tweets['tweetos'] = '' 
    
    #add tweetos first part
    for i in range(len(tweets['text'])):
        try:
            tweets['tweetos'][i] = tweets['text'].str.split(' ')[i][0]
        except AttributeError:    
            tweets['tweetos'][i] = 'other'
    
    #Preprocessing tweetos. select tweetos contains 'RT @'
    for i in range(len(tweets['text'])):
        if tweets['tweetos'].str.contains('@')[i]  == False:
            tweets['tweetos'][i] = 'other'
    
    # remove URLs, RTs, and twitter handles
    for i in range(len(tweets['text'])):
        tweets['text'][i] = " ".join([word for word in tweets['text'][i].split()
                                    if 'http' not in word and '@' not in word and '<' not in word and 'RT' not in word])
    
    #print ( 'FULL TEXT: '  + str(tweets['text'][0]) )
    #print ( 'TWEETOS: ' + str(tweets['tweetos'][0]) )
    
    # save tweets into txt - unnecessary
    #for tweet in tweepy.Cursor(api.search, q='#' + hashtag, count=100, tweet_mode='extended', 
    #    include_entities=True).items(max_tweets):
    #        with open(fName, 'a') as the_file:
    #            the_file.write(str(tweet.full_text) + '\n')
    
    #searched_tweets = [status._json for status in tweepy.Cursor(api.search, q='#' + hashtag, count=100, tweet_mode='extended', 
    #    include_entities=True).items(max_tweets)]
    #print( 'SEARCH RESULT: ' +  str(searched_tweets[10]) )
    
    #json_strings = [json.dumps(json_obj, indent=4, ensure_ascii=False) for json_obj in searched_tweets]
    #json_strings = [json.dumps(json_obj, ensure_ascii=False) for json_obj in searched_tweets]
    #print(json_strings)

    
    # save tweets into json
    #with codecs.open(jsonfName, 'wb', 'utf-8') as jsonfile:
        #json.dump(json_strings, jsonfile)
        #jsonfile.write(json.dumps(json_strings, indent=4, ensure_ascii=False))
    #    jsonfile.write(json.dumps(alltweets, ensure_ascii=False))
    
    #transform the tweepy tweets into a 2D array that will populate the csv	
    #outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]

    #write the csv	
    #with open(csvfName,'w', encoding="utf-8") as f:
    #        writer = csv.writer(f)
    #        writer.writerow(["id","created_at","text"])
    #        writer.writerows(str(tweets['full_text'][0]))
            #f.write("id","created_at","text")
            #f.write(str(tweets['full_text']))
            
            
    # Open/create a file to append data to
    #csvFile = open(csvfName, 'a', encoding='utf-8')
    #Use csv writer
    #csvWriter = csv.writer(csvFile)
    #i = 0
    #while i < len(hashtagList):
    #    for tweet in tweepy.Cursor(api.search, q=hashtagList[0], count=max_tweets, tweet_mode='extended', 
    #                include_entities=True, wait_on_rate_limit=True).items(max_tweets):
    #        # Write a row to the CSV file. I use encode UTF-8
    #        csvWriter.writerow([tweet.created_at, tweet.full_text] )
    #        print (tweet.created_at, tweet.full_text)
    #    i += 1
        
    #csvFile.close()  
    
    wordcloud(tweets,'text')
    
    end = time.time()
    print( "Elapsed time %s" % time.strftime("%H:%M:%S", time.gmtime(end-start)) )
    
    pass

if __name__ == '__main__':
    get_all_tweets()
