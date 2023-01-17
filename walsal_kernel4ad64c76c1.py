from tweepy.streaming import StreamListener

from tweepy import OAuthHandler

import tweepy as tw

import pandas as pd

import re

import jsonpickle

CONSUMER_KEY = 'xxxx'

CONSUMER_SECRET = 'xxxxx'

ACCESS_TOKEN = 'xxxx'

ACCESS_SECRET = 'xxxxx'



# Setup access API

def connect_to_twitter_OAuth():

    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)

    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    

    api = tweepy.API(auth)

    return api

 

# Create API object

api = connect_to_twitter_OAuth()  
query = 'stc'

max_tweets = 1000

lang= 'ar'

tweetCount = 0



tweet_list = []





for tweet in tweepy.Cursor(api.search,q=query,lang=lang).items(max_tweets):         



    #Convert to JSON format

    tweettosave = jsonpickle.encode(tweet._json, unpicklable=False).encode('utf-8')

    tweet_list.append(tweettosave)



    tweetCount += 1



    #Display how many tweets we have collected

    print("Downloaded {0} tweets".format(tweetCount))
tweet_list[0:2]
def tweets_to_df(tweets):

    

   

    

    text = []

    weekday = []

    month = []

    day = []

    hour = []

    hashtag = []

    url = []

    favorite = []

    reply = []

    retweet = []

    follower = []

    following = []

    user = []

    screen_name = []



    for t in tweets:

        t = jsonpickle.decode(t)

        

        # Text

        text.append(t['text'])

        

        # Decompose date

        date = t['created_at']

        weekday.append(date.split(' ')[0])

        month.append(date.split(' ')[1])

        day.append(date.split(' ')[2])

        

        time = date.split(' ')[3].split(':')

        hour.append(time[0]) 

        

        # Has hashtag

        if len(t['entities']['hashtags']) == 0:

            hashtag.append(0)

        else:

            hashtag.append(1)

            

        # Has url

        if len(t['entities']['urls']) == 0:

            url.append(0)

        else:

            url.append(1)

            

        # Number of favs

        favorite.append(t['favorite_count'])

        

        # Is reply?

        if t['in_reply_to_status_id'] == None:

            reply.append(0)

        else:

            reply.append(1)       

        

        # Retweets count

        retweet.append(t['retweet_count'])

        

        # Followers number

        follower.append(t['user']['followers_count'])

        

        # Following number

        following.append(t['user']['friends_count'])

        

        # Add user

        user.append(t['user']['name'])



        # Add screen name

        screen_name.append(t['user']['screen_name'])

        

    d = {'text': text,

         'weekday': weekday,

         'month' : month,

         'day': day,

         'hour' : hour,

         'has_hashtag': hashtag,

         'has_url': url,

         'fav_count': favorite,

         'is_reply': reply,

         'retweet_count': retweet,

         'followers': follower,

         'following' : following,

         'user': user,

         'screen_name' : screen_name

        }

    

    return pd.DataFrame(data = d)

        

tweets_df = tweets_to_df(tweet_list)
#Preprocessing del RT @blablabla:

tweets = tweets_df

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

                                if 'http' not in word and '@' not in word and '<' not in word])





tweets['text'][1]
tweets['text'] = tweets['text'].apply(lambda x: re.sub('[!@#$:).;,?&]', '', x.lower()))

tweets['text'] = tweets['text'].apply(lambda x: re.sub('  ', ' ', x))

tweets['text'][1]
tweets.head()
tweets.to_csv('stcdata.csv')

query = 'mobily'

max_tweets = 1000

lang= 'ar'

tweetCount = 0



tweet_list = []





for tweet in tweepy.Cursor(api.search,q=query,lang=lang).items(max_tweets):         



    #Convert to JSON format

    tweettosave = jsonpickle.encode(tweet._json, unpicklable=False).encode('utf-8')

    tweet_list.append(tweettosave)



    tweetCount += 1



    #Display how many tweets we have collected

    print("Downloaded {0} tweets".format(tweetCount))
tweets_df = tweets_to_df(tweet_list)

#Preprocessing del RT @blablabla:

tweets = tweets_df
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

                                if 'http' not in word and '@' not in word and '<' not in word])





tweets['text'][1]
tweets['text'] = tweets['text'].apply(lambda x: re.sub('[!@#$:).;,?&]', '', x.lower()))

tweets['text'] = tweets['text'].apply(lambda x: re.sub('  ', ' ', x))

tweets['text'][1]
tweets.head()
tweets.to_csv('mobilydata.csv')

query = 'zain'

max_tweets = 1000

lang= 'ar'

tweetCount = 0



tweet_list = []





for tweet in tweepy.Cursor(api.search,q=query,lang=lang).items(max_tweets):         



    #Convert to JSON format

    tweettosave = jsonpickle.encode(tweet._json, unpicklable=False).encode('utf-8')

    tweet_list.append(tweettosave)



    tweetCount += 1



    #Display how many tweets we have collected

    print("Downloaded {0} tweets".format(tweetCount))
tweets_df = tweets_to_df(tweet_list)

#Preprocessing del RT @blablabla:

tweets = tweets_df
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

                                if 'http' not in word and '@' not in word and '<' not in word])





tweets['text'][1]
tweets['text'] = tweets['text'].apply(lambda x: re.sub('[!@#$:).;,?&]', '', x.lower()))

tweets['text'] = tweets['text'].apply(lambda x: re.sub('  ', ' ', x))

tweets['text'][1]
tweets.head()
tweets.to_csv('Zaindata.csv')
