!pip install twitter
import pandas as pd
pd.options.display.max_colwidth = 100
import matplotlib.pyplot as plt
import twitter
disaster_tweets_df = pd.read_csv('../input/disaster-tweets/tweets.csv',
                                 usecols=['keyword', 'location', 'text', 'target'])
disaster_tweets_df
disaster_tweets_df['target'].value_counts()
# Establishing the access to twitter api
# Here I used my twitter credentials (there are now invalidated)
# You will need to plug in yours.

# VERY IMPORTANT!
# Regenerate/revoke your keys, if you decide to publish your version of the notebook

tw={
    'Consumer Key': 'your_consumer_key',
    'Consumer Secret': 'your_consumer_secret',
    'Access Token': 'your_access_token',
    'Access Token Secret': 'your_access_secret',
   }

auth = twitter.oauth.OAuth(tw['Access Token'],
                           tw['Access Token Secret'],
                           tw['Consumer Key'],
                           tw['Consumer Secret'])

twitter_api = twitter.Twitter(auth=auth)
q = 'covid19'
number = 10 # number of tweets to query
search_results = twitter_api.search.tweets(q=q, count=number)
print(search_results.keys())
statuses = search_results['statuses']
example_df = pd.DataFrame(
    data=[[q, s['user']['location'], s['text'], None] for s in statuses],
    columns = ['keyword', 'location', 'text', 'target'],
            )

example_df
# Just as example, here I use four topics.
# Feel free to complement/ammend the list with yours.
keywords=['arsonist', 'lockdown', 'fire', 'crush']
def collect_tweets(keywords, count=10):
    df = pd.DataFrame(columns=['keyword', 'location', 'text', 'target'])
    for q in keywords:
        search_results = twitter_api.search.tweets(q=q, count=count)
        tmp_df = pd.DataFrame(
            data=[[q, s['user']['location'], s['text'], None] for s in search_results['statuses']],
             columns = ['keyword', 'location', 'text', 'target'],
            )
        df = df.append(tmp_df, ignore_index=True)
    
    return df
# We collect 20 tweets in total. 5 tweets over 4 topics.
tweet_collection_df = collect_tweets(keywords, count=5)
tweet_collection_df
tweet_collection_df.to_csv('tweet_collection_df.csv', index=False)
!ls tweet_collection_df.csv -l