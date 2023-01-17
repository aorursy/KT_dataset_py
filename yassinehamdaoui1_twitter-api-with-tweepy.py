import pandas as pd

import tweepy as tw



consumer_key = "oS4qitRipHwUzsjANd2CiFMX5"

consumer_secret ="HBgzw1T0lRA3b64tw2T99aQi6ASQfe2tRWfpTG2zNzMzymhADc"

access_token = "764059813198299136-SzoWF9uPtZePB0fCNWnKnomxgmhVTGC"

access_token_secret ="a1XEyPRBO64h4gC25f9nFmIau83RuxkssHal9PZ0Be53b"



auth = tw.OAuthHandler(consumer_key, consumer_secret)

auth.set_access_token(access_token, access_token_secret)

api = tw.API(auth)



search_words ="earthquake+disaster" # you gonna search by hashtags , the "+" stands for concatenating keywords

date = "2020-01-01" # choose any date with the format "yyyy-mm-dd"

number_posts = 100 # choose the number of posts available in that day 







posts =[]



tweets = tw.Cursor(api.search,q=search_words,lang="en",since=date).items(number_posts)





for tweet in tweets:

    posts.append([tweet.user.screen_name,tweet.user.location,tweet.text])





df = pd.DataFrame(posts)

df.to_csv('data.csv',index=False,header=["username","location","text"])