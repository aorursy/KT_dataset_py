import os
import tweepy
import pandas as pd
from datetime import datetime
import pickle
consumer_key = ''
consumer_secret = ''
access_key = ''
access_secret = ''
#authorize twitter, initialize tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)
def get_all_tweets(screen_name, api):
    '''
    Getting all tweets beyond the limit of tweepy    
    Args-
        screen_name: Screen name (not handles) of the twitter user
        api: The tweepy api
    Return-
        alltweets: List of all the tweets w.r.t the user 
    '''
    #Twitter only allows access to a users most recent 3240 tweets with this method    
    #initialize a list to hold all the tweepy Tweets
    alltweets = []  
    
    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200, tweet_mode='extended',include_rts=True)
    
    #save most recent tweets
    alltweets.extend(new_tweets)
    
    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1
    
    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print(f"getting tweets before {oldest} tweet")
        
        #all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest,tweet_mode='extended')
        
        #save most recent tweets
        alltweets.extend(new_tweets)
        
        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        
        print(f"...{len(alltweets)} tweets downloaded so far")
    
    return alltweets
# Screen names and saving the tweets. 
screen_name_1 = 'bhutanisanyam1' 
screen_name_2 = 'ctdsshow'

tweet_pkl_fldr = "data/tweets"
if not os.path.exists(tweet_pkl_fldr):
    os.makedirs(tweet_pkl_fldr)

tweets_pkl = os.listdir(tweet_pkl_fldr)
kaggle_data_dir = "data/ctds_bundle_archive"
external_data_dir = "data/"
screen_names = [screen_name_1, screen_name_2]
# Added a check so as to not run the tweet getter again! 
for sn in screen_names:
    if sn +'.pkl' not in tweets_pkl:
        print ("Fetching tweet from ", sn)
        all_tweets = get_all_tweets(sn, api)
        print ('')
    
        print ("Saving twitter data as "+ sn + ".pkl" +" pickle file")
        with open(os.path.join(tweet_pkl_fldr, sn +'.pkl'),"wb") as op:
            pickle.dump(all_tweets, op)
            
    print ("Data available..")
    
print ("Done!")
    
tweets_pkl = os.listdir(tweet_pkl_fldr)    
sanyam_tweets = []

with open(os.path.join(tweet_pkl_fldr, tweets_pkl[0]),"rb") as op:
    sanyam_tweets = pickle.load(op)
    print("Number of tweets by Sanyam - ", len(sanyam_tweets))
    
ctds_tweets = []

with open(os.path.join(tweet_pkl_fldr, tweets_pkl[1]),"rb") as op:
    ctds_tweets = pickle.load(op)
    print("Number of tweets by CTDS - ", len(ctds_tweets))
endDate = datetime(2020, 6, 20, 0, 0, 0) #Stopping analysis at June 20th, 2020
startDate = datetime(2019, 7, 20, 0, 0, 0) #Starting analysis at July 20th, 2020

tweets_sanyam = []
for tweet in sanyam_tweets:
    if startDate < tweet.created_at < endDate:
        tweets_sanyam.append(tweet)
        
print ("Tweets that can be considered from Sanyam -", len(tweets_sanyam))

tweets_ctds = []
for tweet in ctds_tweets:
    if startDate < tweet.created_at < endDate:
        tweets_ctds.append(tweet)
        
print ("Tweets that can be considered from CTDS -", len(tweets_ctds))
sanyam_alltweet_list = [[tweet.id_str, tweet.created_at, tweet.full_text, tweet.entities['user_mentions'],
                    tweet.retweet_count, tweet.favorite_count] for tweet in tweets_sanyam]

ctds_alltweet_list = [[tweet.id_str, tweet.created_at, tweet.full_text, tweet.entities['user_mentions'],
                    tweet.retweet_count, tweet.favorite_count] for tweet in tweets_ctds]

sanyam_df = pd.DataFrame(sanyam_alltweet_list)
ctds_df = pd.DataFrame(ctds_alltweet_list)

columns = ["tweet_id", "created_date", "full_text", "user_mentions", "retweet_count", "fav_count"]

sanyam_df.columns = ctds_df.columns = columns
ctds_df.to_csv(os.path.join(external_data_dir, "ctds_all_tweet.csv"), index=False)
sanyam_df.to_csv(os.path.join(external_data_dir, "sanyam_all_tweet.csv"), index=False)
converse = ["talk", "interview","episode"]

ctds_sanyam = [tweet for tweet in tweets_sanyam if any(i in tweet.full_text.lower() for i in converse)]
ctds_ctds = [tweet for tweet in tweets_ctds if any(i in tweet.full_text.lower() for i in converse)]
# sanyam_df['created_date'] = pd.to_datetime(sanyam_df['created_date']).dt.date
# ctds_tweet_df['created_date'] = pd.to_datetime(ctds_tweet_df['created_date'])
print("Tweets on CTDS from Sanyam - ", len(ctds_sanyam))
print("Tweets on CTDS from CTDS - ", len(ctds_ctds))
import re

no_ama_sanyam = [tweet for tweet in ctds_sanyam if not re.search(r"\bama\b", tweet.full_text.lower())]
no_ama_ctds = [tweet for tweet in ctds_ctds if not re.search(r"\bama\b", tweet.full_text.lower())]
print("Tweets inviting AMA from Sanyam - ", len(ctds_sanyam)-len(no_ama_sanyam))
print("Tweets inviting AMA from CTDS - ", len(ctds_ctds)-len(no_ama_ctds))
def ctds_mention(user_mentions):
    '''
    To reduce duplicates, removing the tweets from Sanyam that might have CTDS mentions
    '''
    ctds = [True if i['screen_name'] == 'ctdsshow' else False for i in user_mentions]
    return any(ctds)
no_ctds_sanyam = [i for i in no_ama_sanyam if not ctds_mention(i.entities['user_mentions'])]
print ("Sanyam's tweets on CTDS show with not @ctdsshow mention - ", len(no_ctds_sanyam))
# This might be a mistake!
sanyam_tweet_list = [[tweet.id_str, tweet.created_at, tweet.full_text, tweet.entities['user_mentions'],
                    tweet.retweet_count, tweet.favorite_count] for tweet in no_ama_sanyam]

ctds_tweet_list = [[tweet.id_str, tweet.created_at, tweet.full_text, tweet.entities['user_mentions'],
                    tweet.retweet_count, tweet.favorite_count] for tweet in no_ama_ctds]
# Moving to a df, master df for now
sanyam_tweet_df = pd.DataFrame(sanyam_tweet_list)
ctds_tweet_df = pd.DataFrame(ctds_tweet_list)

columns = ["tweet_id", "created_date", "full_text", "user_mentions", "retweet_count", "fav_count"]

sanyam_tweet_df.columns = ctds_tweet_df.columns = columns
sanyam_tweet_df.shape, ctds_tweet_df.shape
ctds_tweet_df['user_mentions'] = ctds_tweet_df['user_mentions'].apply(lambda x: ','.join([i['screen_name'] for i in x])) 
sanyam_tweet_df['user_mentions'] = sanyam_tweet_df['user_mentions'].apply(lambda x: ','.join([i['screen_name'] for i in x])) 
sanyam_tweet_df.shape, ctds_tweet_df.shape
ctds_tweet_df.to_csv(os.path.join(external_data_dir, "ctds_tweet.csv"), index=False)
sanyam_tweet_df.to_csv(os.path.join(external_data_dir, "sanyam_tweet.csv"), index=False)
import os
import tweepy
import pandas as pd
from datetime import datetime
import pickle
# Reading all the data that is presently available
external_data_dir = "data"
kaggle_data_dir = "data/ctds_bundle_archive"

episode_df = pd.read_csv(os.path.join(kaggle_data_dir, 'Episodes.csv'))
description_df = pd.read_csv(os.path.join(kaggle_data_dir, 'Description.csv'))

ctds_tweet_df = pd.read_csv(os.path.join(external_data_dir, "ctds_tweet.csv"))
sanyam_tweet_df = pd.read_csv(os.path.join(external_data_dir, "sanyam_tweet.csv"))
# df = sanyam_tweet_df[~sanyam_tweet_df["full_text"].str.contains("|".join(converse))]
no_tweet = episode_df[episode_df.heroes_twitter_handle.isnull()]
no_handle_dates = no_tweet['release_date'].values

print ("Number of episodes where there were no twitter handles mentioned - ", no_tweet.shape[0])
# Formatting dataes for filtering
episode_df['release_date'] = pd.to_datetime(episode_df['release_date'])
sanyam_tweet_df['created_date'] = pd.to_datetime(sanyam_tweet_df['created_date'])
ctds_tweet_df['created_date'] = pd.to_datetime(ctds_tweet_df['created_date'])
episode_df['release_date_only'] = episode_df['release_date'].dt.date
# Creating an event date, something like a launch date
sanyam_tweet_df['event_date'] = sanyam_tweet_df['created_date'].dt.date
ctds_tweet_df['event_date'] = ctds_tweet_df['created_date'].dt.date
# Tweepy is sensitive
corrected_handels = {
                     "jfpuget":"JFPuget", "guggersylvain":"GuggerSylvain",\
                     "sanhestpasmoi":"SanhEstPasMoi","arnocandel":"ArnoCandel",\
                     "giba1":"Giba1","stacknet_":"StackNet_","johnmillertx":"JohnMillerTX",\
                     "walterreade":"WalterReade","scitator":"Scitator","mark_a_landry":"Mark_a_Landry",\
                     "madeupmasters":"MadeUpMasters"
                    }
# all_interviewees = episode_df["heroes_twitter_handle"]
# nan_elems = all_interviewees.isnull()
# twitter_users = all_interviewees[~nan_elems]

# all_handles = []
# for i in twitter_users.values:
#     if "|" not in i:
#         all_handles.append(i)
#     else:
#         handles = [i.strip() for i in i.split("|")]
#         all_handles.extend(handles)

# all_handles = [corrected_handels[i] if i in corrected_handels.keys() else i for i in all_handles]
# user_details = api.lookup_users(screen_names=all_handles)
# hero_followers = {i.screen_name:i.followers_count for i in user_details}

# with open("data/hero_followers.pkl","wb") as op:
#     pickle.dump(hero_followers, op)
# Saving it again, do not want to re-do
with open(os.path.join(external_data_dir,"hero_followers.pkl"),"rb") as op:
    hero_followers = pickle.load(op)
def get_total_fcount(hero):
    follower_count = 0
    if not pd.isnull(hero):
        try:
            if "|" not in hero:
                follower_count += hero_followers[hero]     
            else:
                heros = [i.strip() for i in hero.split("|")]
                for hero in heros:
                    try:
                        follower_count += hero_followers[hero]
                    except:
                        if hero in corrected_handels.keys():
                            follower_count += hero_followers[corrected_handels[hero]]
                        else:
                            continue
        except KeyError:
            if hero in corrected_handels.keys():
                follower_count += hero_followers[corrected_handels[hero]]     
            else:
                print (hero)
        
    return follower_count
episode_df['hero_follower_count'] = episode_df["heroes_twitter_handle"].apply(lambda x: get_total_fcount(x))
def separate_str(s):
    if "|" not in s:
        return [s.lower()]
    else:
        print ("more than one interviewee present: ", s)
        return [j.lower().strip() for j in s.split("|")]
    
episode_df['episode_hero'] = [separate_str(i) if isinstance(i,str) else '' for i in episode_df['heroes_twitter_handle'].values.tolist()]
# episode_df['episode_hero'] 
episode_df.shape, sanyam_tweet_df.shape
sanyam_tweet_df['episode_hero_tweet'] = [i.lower().split(",") if isinstance(i,str) else '' for i in sanyam_tweet_df['user_mentions']] 
ctds_tweet_df['episode_hero_tweet'] = [i.split(",") if isinstance(i,str) else '' for i in ctds_tweet_df['user_mentions']] 
sanyam_tweet_df.shape, ctds_tweet_df.shape
# df = sanyam_tweet_df[~sanyam_tweet_df["full_text"].str.contains("|".join(converse))]
def split_to_rows(df, col_name):
    '''Can be done JUST because data is small!'''
    temp_series = df[col_name].apply(pd.Series).reset_index()\
                              .melt(id_vars='index').dropna()[['index', 'value']]\
                              .set_index('index')

    df = pd.merge(temp_series, df, left_index=True, right_index=True)
    del df[col_name]
    df.rename(columns = {'value':col_name}, inplace = True) 

    df.reset_index(inplace=True)
    del df["index"]
    
    return df
episode_df = split_to_rows(episode_df, 'episode_hero')
sanyam_tweet_df = split_to_rows(sanyam_tweet_df, 'episode_hero_tweet')
ctds_tweet_df = split_to_rows(ctds_tweet_df, 'episode_hero_tweet')
sanyam_tweet_df.shape, ctds_tweet_df.shape
# df = sanyam_tweet_df[~sanyam_tweet_df["episode_hero_tweet"].isin(['bhutanisanyam1','ctdsshow'])]
sanyam_tweet_df = sanyam_tweet_df[~sanyam_tweet_df["user_mentions"].isnull()]
sanyam_tweet_df.shape
episode_df["start_date"] = episode_df["release_date_only"] - pd.DateOffset(1)
episode_df["end_date"] = episode_df["release_date_only"] + pd.DateOffset(1)

episode_df["start_date"] = pd.to_datetime(episode_df["start_date"]).dt.date
episode_df["end_date"] = pd.to_datetime(episode_df["end_date"]).dt.date
episode_df = episode_df.assign(key=1)
sanyam_tweet_df = sanyam_tweet_df.assign(key=1)
df_merge = pd.merge(episode_df, sanyam_tweet_df, on='key').drop('key',axis=1)
df_merge = df_merge.query('event_date >= start_date and event_date <= end_date')
df_merge_sanyam = df_merge[df_merge["episode_hero_tweet"].str.lower() ==df_merge["episode_hero"].str.lower()]
df_merge_sanyam.shape
df_merge_sanyam.drop_duplicates(subset=["tweet_id"], keep='first', inplace=True)

agg_columns = ['fav_count','retweet_count']
agg_functions = {i:('first' if i not in agg_columns else 'sum') for i in df_merge_sanyam.columns}

df_merge_sanyam = df_merge_sanyam.groupby(["episode_id","heroes"]).agg(agg_functions)
df_merge_sanyam.shape
episode_df = episode_df.assign(key=1)
ctds_tweet_df = ctds_tweet_df.assign(key=1)
df_merge = pd.merge(episode_df, ctds_tweet_df, on='key').drop('key',axis=1)
df_merge = df_merge.query('event_date >= start_date and event_date <= end_date')
df_merge_ctds = df_merge[df_merge["episode_hero_tweet"].str.lower() ==df_merge["episode_hero"].str.lower()]
df_merge_ctds.drop_duplicates(subset=["tweet_id"], keep='first', inplace=True)

agg_columns = ['fav_count','retweet_count']
agg_functions = {i:('first' if i not in agg_columns else 'sum') for i in df_merge_ctds.columns}

df_merge_ctds = df_merge_ctds.groupby(["episode_id","heroes"]).agg(agg_functions)
df_merge_ctds.shape
df_merge_ctds.to_csv(os.path.join(external_data_dir,"ctds_episode_tweets.csv"),index=None)
df_merge_sanyam.to_csv(os.path.join(external_data_dir,"sanyam_episode_tweets.csv"),index=None)
no_tweet = episode_df[episode_df.heroes_twitter_handle.isnull()]
no_tweet.to_csv(os.path.join(external_data_dir,"no_tweets.csv"),index=None)


