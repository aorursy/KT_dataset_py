# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from langdetect import detect #To detect language

import spacy

import nltk

from textblob import TextBlob

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from nltk.tokenize import RegexpTokenizer



tokenizer = RegexpTokenizer(r'\w+')

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#nlp = spacy.load('en')
tweets_df  = pd.read_csv("../input/pulwama_tweets.csv")

del tweets_df['Unnamed: 0']
cities_data = pd.read_csv('../input/list_of_cities_and_towns_in_india-834j.csv')
cities_data.head()
tweets_df.head()
#x = TextBlob(i[0])

#x = x.translate(to='en')

#tweets_df['text'] = tweets_df['text'].apply(lambda x : TextBlob(x).translate(to='en'))

#sample_df = tweets_df[tweets_df['followers']>40000]
#sample_df.shape
#ent = []

#for doc in nlp.pipe(sample_df['text'], batch_size=10000, n_threads=3):

#    ent.append(doc.ents)

#    pass
#ent[10:20]
#nouns = []

#for doc in nlp.pipe(sample_df['text'], batch_size=10000, n_threads=3):

#    nouns.append(list(doc.noun_chunks))
#nouns[100:200]
tweets_df.head()
#tweets_df.fillna("india",inplace = True)

locations = tweets_df['location'].unique()

print(len(locations))
#raw['location'].nunique()
tweets_df['text'].fillna('pulwama attack',inplace = True)

tweets_df['text'] = tweets_df['text'].apply(lambda x : 'Pulwama attack' if x.startswith('http') else x)

tweets_df['location'].fillna('india',inplace = True)

tweets_df['location'] = tweets_df['location'].apply(lambda x : x.lower())

tweets_df['location'] = tweets_df['location'].apply(lambda x : 'india' if x.endswith('भारत') else x)

tweets_df['location'] = tweets_df['location'].apply(lambda x : 'india' if x.startswith('भारत') else x)

tweets_df['location'] = tweets_df['location'].apply(lambda x : 'india' if x.startswith('हिन्दू') else x)

tweets_df['location'] = tweets_df['location'].apply(lambda x : x.replace(',',' '))

tweets_df['location'] = tweets_df['location'].apply(lambda x : x[1:]  if x.startswith(' ') else x)

tweets_df['location'] = tweets_df['location'].apply(lambda x : x[0:-1]  if x.endswith(' ') else x)

tweets_df['location'] = tweets_df['location'].apply(lambda x: x.replace('india',''))
cities_data['Name of City'].dropna(axis = 0,inplace = True)

cities_data['Name of City'] = cities_data['Name of City'].apply(lambda x : x.lower())

city = cities_data['Name of City']

city = city[0:-3]
city[0:100]
def findloc(x):

    resolved_name = "None"

    

    x = tokenizer.tokenize(x)

    #print(x)

    for i in x:

        #print(i)

        for j in city:

            #print(j)

            city_tokens = tokenizer.tokenize(j)

            #print(i,j)

            if( i in city_tokens): 

                resolved_name  = j

                    



    return resolved_name
tweets_df['new_location'] = tweets_df['location'].apply(findloc)
tweets_df['new_location'] = tweets_df['location'].apply(findloc)
tweets_df.isnull().sum()
tweets_df['new_location'].describe()
tweets_df = tweets_df[tweets_df['new_location']!='None']
tweets_df.reset_index()

tweets_df.to_csv('tweets_location.csv',index = False)
sample_df.head()
def analize_sentiment(tweet):

    '''

    Utility function to classify the polarity of a tweet

    using textblob.

    '''

    analysis = TextBlob(clean_tweet(tweet))

    if analysis.sentiment.polarity > 0:

        return 1

    elif analysis.sentiment.polarity == 0:

        return 0

    else:

        return -1
#sample_df.reset_index(inplace = True)
#sample_df['SA'] = np.array([ analize_sentiment(tweet) for tweet in sample_df['text'] ])

#sample_df['SA'].value_counts()
#list(pos['text'][10:20])
#pos_tweets = [ tweet for index, tweet in enumerate(sample_df['text']) if sample_df['SA'][index] > 0]

#neu_tweets = [ tweet for index, tweet in enumerate(sample_df['text']) if sample_df['SA'][index] == 0]

#neg_tweets = [ tweet for index, tweet in enumerate(sample_df['text']) if sample_df['SA'][index] < 0]
#sample_df.head()
#location = sample_df[['location','SA']]
#location[200:300]
#from textblob import TextBlob

import re



def clean_tweet(tweet):

    '''

    Utility function to clean the text in a tweet by removing 

    links and special characters using regex.

    '''

    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
#tweets_df['text'] = tweets_df['text'].apply(clean_tweet)
st = 'पाकिस्तान के खिलाफ डिबेट में अगर कोई कहता'
#st.translate('en')
#tweets_df['text'][3].ipynb_checkpoints/
#tweets_df['language'] = tweets_df['text'].apply(lambda x : detect(x) )
#tweets_df['language'] = 'None'
#tweets_df = tweets_df[tweets_df['text']!='']
#tweets_df.head()
#for i,x in enumerate(tweets_df['text']):

#    try:

#        tweets_df.loc[i]['language'] = detect(x)

#    except:

#        tweets_df.loc[i]['language'] = 'None'
#for i,x in enumerate(tweets_df['text']):

#    tweets_df.iloc[i]['language'] = 'ed'
