# regular expression operations

import re    

# string operation 

import string  

# shuffle the list

from random import shuffle



# linear algebra

import numpy as np 

# data processing

import pandas as pd 



# NLP library

import nltk

# download twitter dataset

from nltk.corpus import twitter_samples                          



# module for stop words that come with NLTK

from nltk.corpus import stopwords          

# module for stemming

from nltk.stem import PorterStemmer        

# module for tokenizing strings

from nltk.tokenize import TweetTokenizer   



# scikit model selection

from sklearn.model_selection import train_test_split



# smart progressor meter

from tqdm import tqdm
# Download the twitter sample data from NLTK repository

nltk.download('twitter_samples')
# read the positive and negative tweets

pos_tweets = twitter_samples.strings('positive_tweets.json')

neg_tweets = twitter_samples.strings('negative_tweets.json')

print(f"positive sentiment 👍 total samples {len(pos_tweets)} \nnegative sentiment 👎 total samples {len(neg_tweets)}")
# Let's have a look at the data

no_of_tweets = 3

print(f"Let's take a look at first {no_of_tweets} sample tweets:\n")

print("Example of Positive tweets:")

print('\n'.join(pos_tweets[:no_of_tweets]))

print("\nExample of Negative tweets:")

print('\n'.join(neg_tweets[:no_of_tweets]))
# helper class for doing preprocessing

class Twitter_Preprocess():

    

    def __init__(self):

        # instantiate tokenizer class

        self.tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,

                                       reduce_len=True)

        # get the english stopwords 

        self.stopwords_en = stopwords.words('english') 

        # get the english punctuation

        self.punctuation_en = string.punctuation

        # Instantiate stemmer object

        self.stemmer = PorterStemmer() 

        

    def __remove_unwanted_characters__(self, tweet):

        

        # remove retweet style text "RT"

        tweet = re.sub(r'^RT[\s]+', '', tweet)



        # remove hyperlinks

        tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

     

        # remove hashtags

        tweet = re.sub(r'#', '', tweet)

        

        #remove email address

        tweet = re.sub('\S+@\S+', '', tweet)

        

        # remove numbers

        tweet = re.sub(r'\d+', '', tweet)

        

        ## return removed text

        return tweet

    

    def __tokenize_tweet__(self, tweet):        

        # tokenize tweets

        return self.tokenizer.tokenize(tweet)

    

    def __remove_stopwords__(self, tweet_tokens):

        # remove stopwords

        tweets_clean = []



        for word in tweet_tokens:

            if (word not in self.stopwords_en and  # remove stopwords

                word not in self.punctuation_en):  # remove punctuation

                tweets_clean.append(word)

        return tweets_clean

    

    def __text_stemming__(self,tweet_tokens):

        # store the stemmed word

        tweets_stem = [] 



        for word in tweet_tokens:

            # stemming word

            stem_word = self.stemmer.stem(word)  

            tweets_stem.append(stem_word)

        return tweets_stem

    

    def preprocess(self, tweets):

        tweets_processed = []

        for _, tweet in tqdm(enumerate(tweets)):        

            # apply removing unwated characters and remove style of retweet, URL

            tweet = self.__remove_unwanted_characters__(tweet)            

            # apply nltk tokenizer

            tweet_tokens = self.__tokenize_tweet__(tweet)            

            # apply stop words removal

            tweet_clean = self.__remove_stopwords__(tweet_tokens)

            # apply stemmer 

            tweet_stems = self.__text_stemming__(tweet_clean)

            tweets_processed.extend([tweet_stems])

        return tweets_processed
# initilize the text preprocessor class object

twitter_text_processor = Twitter_Preprocess()



# process the positive and negative tweets

processed_pos_tweets = twitter_text_processor.preprocess(pos_tweets)

processed_neg_tweets = twitter_text_processor.preprocess(neg_tweets)
pos_tweets[:no_of_tweets], processed_pos_tweets[:no_of_tweets]
# BOW frequency represent the (word, y) and frequency of y class

def build_bow_dict(tweets, labels):

    freq = {}

    ## create zip of tweets and labels

    for tweet, label in list(zip(tweets, labels)):

        for word in tweet:

            freq[(word, label)] = freq.get((word, label), 0) + 1

        

    return freq
# create labels of the tweets

# 1 for positive labels and 0 for negative labels

labels = [1 for i in range(len(processed_pos_tweets))]

labels.extend([0 for i in range(len(processed_neg_tweets))])



# combine the positive and negative tweets

twitter_processed_corpus = processed_pos_tweets + processed_neg_tweets



# build Bog of words frequency 

bow_word_frequency = build_bow_dict(twitter_processed_corpus, labels)
# extract feature for tweet

def extract_features(processed_tweet, bow_word_frequency):

    # feature array

    features = np.zeros((1,3))

    # bias term added in the 0th index

    features[0,0] = 1

    

    # iterate processed_tweet

    for word in processed_tweet:

        # get the positive frequency of the word

        features[0,1] = bow_word_frequency.get((word, 1), 0)

        # get the negative frequency of the word

        features[0,2] = bow_word_frequency.get((word, 0), 0)

    

    return features
# shuffle the positive and negative tweets

shuffle(processed_pos_tweets)

shuffle(processed_neg_tweets)



# create positive and negative labels

positive_tweet_label = [1 for i in processed_pos_tweets]

negative_tweet_label = [0 for i in processed_neg_tweets]



# create dataframe

tweet_df = pd.DataFrame(list(zip(twitter_processed_corpus, positive_tweet_label+negative_tweet_label)), columns=["processed_tweet", "label"])
# train and test split

train_X_tweet, test_X_tweet, train_Y, test_Y = train_test_split(tweet_df["processed_tweet"], tweet_df["label"], test_size = 0.20, stratify=tweet_df["label"])

print(f"train_X_tweet {train_X_tweet.shape}, test_X_tweet {test_X_tweet.shape}, train_Y {train_Y.shape}, test_Y {test_Y.shape}")
# train X feature dimension

train_X = np.zeros((len(train_X_tweet), 3))



for index, tweet in enumerate(train_X_tweet):

    train_X[index, :] = extract_features(tweet, bow_word_frequency)



# test X feature dimension

test_X = np.zeros((len(test_X_tweet), 3))



for index, tweet in enumerate(test_X_tweet):

    test_X[index, :] = extract_features(tweet, bow_word_frequency)



print(f"train_X {train_X.shape}, test_X {test_X.shape}")
train_X[0:5]
def sigmoid(z): 

    

    # calculate the sigmoid of z

    h = 1 / (1+ np.exp(-z))

    

    return h
# implementation of gradient descent algorithm  



def gradientDescent(x, y, theta, alpha, num_iters):



    # get the number of samples in the training

    m = x.shape[0]

    

    for i in range(0, num_iters):

        

        # find linear regression equation value, X and theta

        z = np.dot(x, theta)

        

        # get the sigmoid of z

        h = sigmoid(z)

 

        # calculate the cost function, log loss

        J = (-1/m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1-h)))



        # update the weights theta

        theta = theta - (alpha / m) * np.dot((x.T), (h - y))

   

    J = float(J)

    return J, theta
# set the seed in numpy

np.random.seed(1)

# Apply gradient descent of logistic regression

J, theta = gradientDescent(train_X, np.array(train_Y).reshape(-1,1), np.zeros((3, 1)), 1e-7, 1000)

print(f"The cost after training is {J:.8f}.")

print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")
# predict for the features from learned theata values

def predict_tweet(x, theta):

    

    # make the prediction for x with learned theta values

    y_pred = sigmoid(np.dot(x, theta))

    

    return y_pred
# predict for the test sample with the learned weights for logistics regression

predicted_probs = predict_tweet(test_X, theta)

# assign the probability threshold to class

predicted_labels = np.where(predicted_probs > 0.5, 1, 0)

# calculate the accuracy

print(f"Accuracy is {len(predicted_labels[predicted_labels == np.array(test_Y).reshape(-1,1)]) / len(test_Y)*100:.2f}")