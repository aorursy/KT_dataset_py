# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
t1 = pd.read_csv("../input/russian-troll-tweets/IRAhandle_tweets_1.csv")

t2 = pd.read_csv("../input/russian-troll-tweets/IRAhandle_tweets_2.csv")

t3 = pd.read_csv("../input/russian-troll-tweets/IRAhandle_tweets_3.csv")

t4 = pd.read_csv("../input/russian-troll-tweets/IRAhandle_tweets_4.csv")

t5 = pd.read_csv("../input/russian-troll-tweets/IRAhandle_tweets_5.csv")

t6 = pd.read_csv("../input/russian-troll-tweets/IRAhandle_tweets_6.csv")

t7 = pd.read_csv("../input/russian-troll-tweets/IRAhandle_tweets_7.csv")

t8 = pd.read_csv("../input/russian-troll-tweets/IRAhandle_tweets_8.csv")

t9 = pd.read_csv("../input/russian-troll-tweets/IRAhandle_tweets_9.csv")

frames = [t1,t2,t3,t4,t5,t6,t7,t8,t9]

tweets = pd.concat(frames)
# We can print out the titles of all the columns, this can help us organize our information

print(list(tweets.columns))
print(tweets)
tweets
tweets = tweets[tweets['publish_date'].str.contains("2016")] 



tweets
tweets = tweets[tweets['language'].str.contains("English")] 



tweets
tweets = tweets[tweets['content'].str.contains("Trump")] 



tweets
tweets = tweets.drop(['external_author_id', 'post_type','updates','new_june_2018','retweet','harvested_date'], axis=1)



tweets
tweets = tweets[['content']]



tweets
#iloc is panda specific

rows = tweets.iloc[0]['content'].split()

print(rows)
# Declare a dictionary

mainDict= {}



for index, row in tweets.iterrows(): 

    rows = row["content"].split()

    

    for w in rows:

        if w in mainDict:

            mainDict[w] = mainDict[w] + rows.count(w)

        else:

            mainDict[w] = rows.count(w)



print (mainDict)    
print(len(mainDict))



wordDel = ['in','you','the','The','I','is','on','a','A','to','To','of','for','and','it']

for w in wordDel:

    mainDict.pop(w)



wordDel2 = ['will','with','that','this','be','at','as','he','his','by','not','they']

for x in wordDel2:

    mainDict.pop(x)



wordDel3 = ['-','&','about','has','from','was','have','who','all','say','my','out']

for z in wordDel3:

    mainDict.pop(z)



wordDel4 = ['says','up','but','we','like','are','if','our','via','�','This','your','an']

for y in wordDel4:

    mainDict.pop(y)

    

print (len(mainDict))
# We can sort the dictionary in descending order

mainDict = {k: v for k, v in sorted(mainDict.items(), key=lambda item: item[1],reverse=True)}



counter=0

for k, v in mainDict.items():

    if counter > 19:

        break

    print(v, "\t", k)

    counter += 1





# You can also turn the mainDict into a list with just the keys and print the top 20

topTwenty = list(mainDict.keys())

print(topTwenty[:20])