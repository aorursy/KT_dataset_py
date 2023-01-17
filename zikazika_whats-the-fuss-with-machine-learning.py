!pip install tweepy
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns

import re

import time

import string

import warnings



# for all NLP related operations on text

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.stem import WordNetLemmatizer

from nltk.stem.porter import *

from nltk.classify import NaiveBayesClassifier

from wordcloud import WordCloud



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB





# To consume Twitter's API

import tweepy

from tweepy import OAuthHandler 



# To identify the sentiment of text

from textblob import TextBlob

from textblob.sentiments import NaiveBayesAnalyzer

from textblob.np_extractors import ConllExtractor



# ignoring all the warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)



# downloading stopwords corpus

nltk.download('stopwords')

nltk.download('wordnet')

nltk.download('vader_lexicon')

nltk.download('averaged_perceptron_tagger')

nltk.download('movie_reviews')

nltk.download('punkt')

nltk.download('conll2000')

nltk.download('brown')

stopwords = set(stopwords.words("english"))



# for showing all the plots inline

%matplotlib inline
# keys and tokens to access Twitter API

consumer_key = 'Sec3MvclRIx2RVlgu9l0SJX6D'

consumer_secret = 'ayoPNWtBm7fWpMBoK6EwRmegu3SW8Rw9mzJkottkv97quPe941'

access_token = '736550752760406018-so5CPJrEbJKb3c3Pq8va3VFr0yk4S0E'

access_token_secret = 'Cgr8tz0h6FTU7kxAjDzpHnjffNTHxWsBytXnu4Ihd1TFb'
class TwitterClient(object): 

    def __init__(self): 

        #Initialization method. 

        try: 

            # create OAuthHandler object 

            auth = OAuthHandler(consumer_key, consumer_secret) 

            # set access token and secret 

            auth.set_access_token(access_token, access_token_secret) 

            # create tweepy API object to fetch tweets 

            # add hyper parameter 'proxy' if executing from behind proxy "proxy='http://172.22.218.218:8085'"

            self.api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

            

        except tweepy.TweepError as e:

            print(f"Error: Tweeter Authentication Failed - \n{str(e)}")



    def get_tweets(self, query, maxTweets = 1000):

        #Function to fetch tweets. 

        # empty list to store parsed tweets 

        tweets = [] 

        sinceId = None

        max_id = -1

        tweetCount = 0

        tweetsPerQry = 100



        while tweetCount < maxTweets:

            try:

                if (max_id <= 0):

                    if (not sinceId):

                        new_tweets = self.api.search(q=query, count=tweetsPerQry)

                    else:

                        new_tweets = self.api.search(q=query, count=tweetsPerQry,

                                                since_id=sinceId)

                else:

                    if (not sinceId):

                        new_tweets = self.api.search(q=query, count=tweetsPerQry,

                                                max_id=str(max_id - 1))

                    else:

                        new_tweets = self.api.search(q=query, count=tweetsPerQry,

                                                max_id=str(max_id - 1),

                                                since_id=sinceId)

                if not new_tweets:

                    print("No more tweets found")

                    break



                for tweet in new_tweets:

                    parsed_tweet = {} 

                    parsed_tweet['tweets'] = tweet.text 



                    # appending parsed tweet to tweets list 

                    if tweet.retweet_count > 0: 

                        # if tweet has retweets, ensure that it is appended only once 

                        if parsed_tweet not in tweets: 

                            tweets.append(parsed_tweet) 

                    else: 

                        tweets.append(parsed_tweet) 

                        

                tweetCount += len(new_tweets)

                print("Downloaded {0} tweets".format(tweetCount))

                max_id = new_tweets[-1].id



            except tweepy.TweepError as e:

                # Just exit if any error

                print("Tweepy error : " + str(e))

                break

        

        return pd.DataFrame(tweets)
twitter_client = TwitterClient()



# calling function to get tweets

tweets_df = twitter_client.get_tweets('Machine Learning', maxTweets=10000)

print(f'tweets_df Shape - {tweets_df.shape}')

tweets_df.head(10)
def underlying_atmosphere_textblob(text):

    analysis = TextBlob(text)

    if analysis.sentiment.polarity > 0: 

        return 'positive'

    elif analysis.sentiment.polarity == 0: 

        return 'neutral'

    else: 

        return 'negative'
sentiments_using_textblob = tweets_df.tweets.apply(lambda tweet: underlying_atmosphere_textblob(tweet))

pd.DataFrame(sentiments_using_textblob.value_counts())
tweets_df['sentiment'] = sentiments_using_textblob

tweets_df.head()
def remove_pattern(text, pattern_regex):

    r = re.findall(pattern_regex, text)

    for i in r:

        text = re.sub(i, '', text)

    

    return text 
# We are keeping cleaned tweets in a new column called 'tidy_tweets'

tweets_df['tidy_tweets'] = np.vectorize(remove_pattern)(tweets_df['tweets'], "@[\w]*: | *RT*")

tweets_df.head(10)
cleaned_tweets = []



for index, row in tweets_df.iterrows():

    # Here we are filtering out all the words that contains link

    words_without_links = [word for word in row.tidy_tweets.split() if 'http' not in word]

    cleaned_tweets.append(' '.join(words_without_links))



tweets_df['tidy_tweets'] = cleaned_tweets

tweets_df.head(10)
tweets_df = tweets_df[tweets_df['tidy_tweets']!='']

tweets_df.head()
tweets_df.drop_duplicates(subset=['tidy_tweets'], keep=False)

tweets_df.head()
tweets_df = tweets_df.reset_index(drop=True)

tweets_df.head()
tweets_df['absolute_tidy_tweets'] = tweets_df['tidy_tweets'].str.replace("[^a-zA-Z# ]", "")
stopwords_set = set(stopwords)

cleaned_tweets = []



for index, row in tweets_df.iterrows():

    

    # filerting out all the stopwords 

    words_without_stopwords = [word for word in row.absolute_tidy_tweets.split() if not word in stopwords_set and '#' not in word.lower()]

    

    # finally creating tweets list of tuples containing stopwords(list) and sentimentType 

    cleaned_tweets.append(' '.join(words_without_stopwords))

    

tweets_df['absolute_tidy_tweets'] = cleaned_tweets

tweets_df.head(10)
tokenized_tweet = tweets_df['absolute_tidy_tweets'].apply(lambda x: x.split())

tokenized_tweet.head()
word_lemmatizer = WordNetLemmatizer()



tokenized_tweet = tokenized_tweet.apply(lambda x: [word_lemmatizer.lemmatize(i) for i in x])

tokenized_tweet.head()
for i, tokens in enumerate(tokenized_tweet):

    tokenized_tweet[i] = ' '.join(tokens)



tweets_df['absolute_tidy_tweets'] = tokenized_tweet

tweets_df.head(10)
textblob_key_phrases = []

extractor = ConllExtractor()



for index, row in tweets_df.iterrows():

    # filerting out all the hashtags

    words_without_hash = [word for word in row.tidy_tweets.split() if '#' not in word.lower()]

    

    hash_removed_sentence = ' '.join(words_without_hash)

    

    blob = TextBlob(hash_removed_sentence, np_extractor=extractor)

    textblob_key_phrases.append(list(blob.noun_phrases))



textblob_key_phrases[:10]
tweets_df['key_phrases'] = textblob_key_phrases

tweets_df.head(10)
def generate_wordcloud(all_words):

    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=100, relative_scaling=0.5, colormap='Dark2').generate(all_words)



    plt.figure(figsize=(14, 10))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis('off')

    plt.show()
all_words = ' '.join([text for text in tweets_df['absolute_tidy_tweets'][tweets_df.sentiment == 'positive']])

generate_wordcloud(all_words)
all_words = ' '.join([text for text in tweets_df['absolute_tidy_tweets'][tweets_df.sentiment == 'negative']])

generate_wordcloud(all_words)
all_words = ' '.join([text for text in tweets_df['absolute_tidy_tweets'][tweets_df.sentiment == 'neutral']])

generate_wordcloud(all_words)
# function to collect hashtags

def hashtag_extract(text_list):

    hashtags = []

    # Loop over the words in the tweet

    for text in text_list:

        ht = re.findall(r"#(\w+)", text)

        hashtags.append(ht)



    return hashtags



def generate_hashtag_freqdist(hashtags):

    a = nltk.FreqDist(hashtags)

    d = pd.DataFrame({'Hashtag': list(a.keys()),

                      'Count': list(a.values())})

    # selecting top 15 most frequent hashtags     

    d = d.nlargest(columns="Count", n = 25)

    plt.figure(figsize=(16,7))

    ax = sns.barplot(data=d, x= "Hashtag", y = "Count")

    plt.xticks(rotation=80)

    ax.set(ylabel = 'Count')

    plt.show()
hashtags = hashtag_extract(tweets_df['tidy_tweets'])

hashtags = sum(hashtags, [])
generate_hashtag_freqdist(hashtags)


tweets_df_full_key_phrases = tweets_df[tweets_df['key_phrases'].str.len()>0]
# TF-IDF

tfidf_word_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, stop_words='english')

tfidf_word_feature = tfidf_word_vectorizer.fit_transform(tweets_df_full_key_phrases['tidy_tweets'])
phrase_sents = tweets_df_full_key_phrases['key_phrases'].apply(lambda x: ' '.join(x))



# TF-IDF for key phrases

tfidf_phrase_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2)

tfidf_phrase_feature = tfidf_phrase_vectorizer.fit_transform(phrase_sents)
target_variable = tweets_df_full_key_phrases['sentiment'].apply(lambda x: 0 if x=="negative" else (1 if x=="neutral" else 2))
def naive_model(X_train, X_test, y_train, y_test):

    naive_classifier = LogisticRegression()

    naive_classifier.fit(X_train.toarray(), y_train)



    # predictions over test set

    predictions = naive_classifier.predict(X_test.toarray())



    # calculating Accuracy Score

    print(f'Accuracy Score - {accuracy_score(y_test, predictions)}')

    
X_train, X_test, y_train, y_test = train_test_split(tfidf_word_feature, target_variable, test_size=0.2, random_state=123)

naive_model(X_train, X_test, y_train, y_test)