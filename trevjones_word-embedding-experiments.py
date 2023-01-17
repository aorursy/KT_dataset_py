import pandas as pd 

import numpy as np

import re

import string

from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
# open and read the GloVe twitter dataset into a word2vec dictionary.

# if not on kaggle, the dataset can be acquired from 

# http://nlp.stanford.edu/data/glove.twitter.27B.zip

#

# we found we had the best results when using 50d vectors, as going 

# higher didn't increase accuracy but had a large performance cost



with open("/kaggle/input/glove-global-vectors-for-word-representation/glove.twitter.27B.50d.txt", "rb") as lines:

    w2v = {line.split()[0].decode("utf-8"): np.array([float(value) for value in line.split()[1:]])

           for line in lines}
# read in the csv data

train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

sample = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
# drop the null entries

train[train['text'].isna()]

train.drop(314, inplace = True)

# this function replaces all sequential occurances of a symbol with a single word

def replace_symbol_word(text, symbol, word):

    starIdx = text.find(symbol)

    count = 0

    while starIdx > -1 and count < 20:

        firstIdx = starIdx

        while(starIdx+1 < len(text) and text[starIdx+1] == symbol):

            starIdx += 1

        text = text[:firstIdx] + " " + word + " " + text[starIdx+1:]

        starIdx = -1

        starIdx = text.find(symbol)

        count += 1

    

    return text



# cleans the text by removing urls, numbers, punctuation, and changing any sequence of 3 or more of the same letter to just 2

def clean_text(text):    

    text = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', text)

    

    # remove the characters [\], ['], [`], and ["]

    text = re.sub(r"\\", "", text)

    text = re.sub(r"\'", "", text)

    text = re.sub(r"\`", "", text)

    text = re.sub(r"\"", "", text)

    

    # remove numbers

    text = re.sub(r"[0-9]+", "", text)

    

    # convert text to lowercase

    text = text.strip().lower()

    

    # we attempted to replace symbols with words, but it made performance worse.

    # we tried a few different words, but couldn't find one that worked well with word embeddings

#     text = replace_symbol_word(text, '*', 'abusive')

#     text = replace_symbol_word(text, '!', 'exclaim')

    

    # replace 3 or more of the same letter with just 2

    # no word in the english dictionary has 3 of the same letter in a row,

    # they all use a hyphen in between.

    # for example turns cooooool into cool and yummmmy into yummy

    # doesn't work great for examples like looooool -> lool, but at least 

    # looool with any number of zeros always ends up as the same word lool

    # which helps when doing anything with word counts

    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    

    # replace punctuation characters with spaces

    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

    translate_dict = dict((c, " ") for c in filters)

    translate_map = str.maketrans(translate_dict)

    text = text.translate(translate_map)

    text = ' '.join(text.split())

    

    return text
# clean the text and selected_text and then write them to their own columns

train['clean_selected_text'] = train['selected_text'].apply(clean_text)

train['clean_text'] = train['text'].apply(clean_text)



# X_train = train.copy()



# split the dataset for validation purposes

# make sure to train on the full dataset for submission

X_train, X_val = train_test_split(

    train, train_size = 0.80, random_state = 0)

X_train = X_train.copy()

X_val = X_val.copy()





X_train.head()
# splits up the training data based on sentiment

pos_train = X_train[X_train['sentiment'] == 'positive']

neutral_train = X_train[X_train['sentiment'] == 'neutral']

neg_train = X_train[X_train['sentiment'] == 'negative']



# count vectorizer to get word counts

n = 1

cv = CountVectorizer(ngram_range=(n, n), max_df=0.8, min_df=2,

                                         max_features=None,

                                         stop_words='english')



# vectorize the cleaned selected text for all the data, and for each sentiment

X_train_cv = cv.fit_transform(X_train['clean_selected_text'])



X_pos = cv.transform(pos_train['clean_selected_text'])

X_neutral = cv.transform(neutral_train['clean_selected_text'])

X_neg = cv.transform(neg_train['clean_selected_text'])



# create a dataframe where the columns are all of the words minus stopwords,

# and each row is a tweet in count vectorized form, 

# where each value is the number of times that columns words appears in the tweet

pos_count_df = pd.DataFrame(X_pos.toarray(), columns=cv.get_feature_names())

neutral_count_df = pd.DataFrame(X_neutral.toarray(), columns=cv.get_feature_names())

neg_count_df = pd.DataFrame(X_neg.toarray(), columns=cv.get_feature_names())



# empty dictionaries that we will fill

# these 3 contain the total counts of each word over each sentiment

pos_words = {}

neut_words = {}

neg_words = {}

# these 3 contain the proportion of tweets in sentiment which contain the word

pos_words_proportion = {}

neutral_words_proportion = {}

neg_words_proportion = {}



for k in cv.get_feature_names():

    # gets raw word count of each word for each sentiment

    pos_words[k] = pos_count_df[k].sum()

    neut_words[k] = neutral_count_df[k].sum()

    neg_words[k] = neg_count_df[k].sum()

    

    # divide word counts by number of samples to get proportion

    pos_words_proportion[k] = pos_words[k]/pos_train.shape[0]

    neutral_words_proportion[k] = neut_words[k]/neutral_train.shape[0]

    neg_words_proportion[k] = neg_words[k]/neg_train.shape[0]

    

neg_words_adj = {}

pos_words_adj = {}

neutral_words_adj = {}



# adjust the proportion value to take into account the fact that words will show up in tweets of other sentiments

for key, value in neg_words_proportion.items():

    neg_words_adj[key] = neg_words_proportion[key] - (neutral_words_proportion[key] + pos_words_proportion[key])



for key, value in pos_words_proportion.items():

    pos_words_adj[key] = pos_words_proportion[key] - (neutral_words_proportion[key] + neg_words_proportion[key])



for key, value in neutral_words_proportion.items():

    neutral_words_adj[key] = neutral_words_proportion[key] - (neg_words_proportion[key] + pos_words_proportion[key])
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
class MeanEmbeddingVectorizer(object):

    def __init__(self, word2vec, pos_words, neg_words, neut_words):

        self.pos_words = pos_words

        self.neg_words = neg_words

        self.neut_words = neut_words

        self.word2vec = word2vec

        # if a text is empty we should return a vector of zeros

        # with the same dimensionality as all the other vectors

        self.dim = len(next(iter(word2vec.items()))[1])



    # Here X is the clean_selected_text ground truth, and y is the sentiment.

    def fit(self, X, y):

        ratio = 0.8

        self.average_positive = self.get_average_vector(X[y == 'positive'], 'positive', ratio)

        self.average_neutral = self.get_average_vector(X[y == 'neutral'], 'neutral', ratio)

        self.average_negative = self.get_average_vector(X[y == 'negative'], 'negative', ratio)

        

        # print the similarity between the average negative and positive tweet

        print(np.dot(self.average_negative, self.average_positive)/(np.linalg.norm(self.average_negative)*np.linalg.norm(self.average_positive)))

        return self

    

    # takes data from one sentiment, that sentiment, and the ratio to use for determing which words are often enough to use

    # and then calculates the average vector for that sentiment from that data

    # note the data that gets passed in is from clean_selected_text

    def get_average_vector(self, X, sentiment, ratio):

        # used to get the ratio of how many of that words appears in the given sentiment

        numerator_dict = (self.pos_words if sentiment == 'positive' else self.neg_words if sentiment == 'negative' else self.neut_words)

        denominator_dict = {k: self.pos_words[k] + self.neut_words[k] + self.neg_words[k] for k in self.neut_words.keys()}

        # default dict to handle words we haven't seen and stop words that won't show up in this dict, but do in the clean text

        word_proportion_dict = defaultdict(float)

        for k in numerator_dict.keys():

            word_proportion_dict[k] = numerator_dict[k]/denominator_dict[k]

                

        sent_vec_list = []

        for sent in X:

            sent_word_vecs = []

            for w in sent.split(" "):

                if w in self.word2vec and word_proportion_dict[w] > ratio:

                    # if we have a vector for the word and its ratio is high enough to use for the vector, then add it

                    sent_word_vecs.append(self.word2vec[w])

            if(len(sent_word_vecs) > 0):

                # once we have added all words, if we have at least 1, then get 

                # the average of that tweet and append it to our tweet list

                sent_vec_list.append(np.mean(sent_word_vecs, axis=0))

        

        # return the average of all the tweets over axis 0, so we get one 50d vector that is the average

        # of all the words that appear often in that sentiment's selected_text

        #

        # this means that words that appear often are included multiple times, and thus have more effect

        # which is why we don't weight this with word counts.

        return np.mean(np.array(sent_vec_list), axis=0)



    # transforms one sentence to a vector with the mean of the words

    # sent is a list of words, where each item is one word, this means no need to split here

    def transform(self, sent, sentiment):

        sent_vec_list = []

        scalars = pos_words_adj if sentiment == 'positive' else neg_words_adj

        # checking if its in pos_words allows us to strip all of the stop words that were removed as part of CountVectorizer

        # as pos_words only contains words from the CountVectorizer, and contains all the same words as neg_words and neut_words

        sent_word_vecs = [[x * scalars[w] for x in self.word2vec[w]]  for w in sent if (w in self.word2vec and w in pos_words.keys())]

        if(len(sent_word_vecs) > 0):

            # as long as we have at least 1 word, then average all the words and add it to our list

            sent_vec_list.append(np.mean(sent_word_vecs, axis=0))

        

        # make sure we actually got a vector output, and if we did then return it, otherwise return a vector of zeros

        if(len(sent_vec_list)):

            return np.array(sent_vec_list)

        return np.zeros(self.dim)

    

    # get the cosine similarity between the 3 average vectors and a given sentence

    def get_sent_dist(self, sent, sentiment):

        sent_vect = self.transform(sent, sentiment)

                     

        # the sum will be zero if we return an array of zeros from transform becaue we couldn't find any valid words

        if sent_vect.sum() != 0.0:

            # cosine similarity = dot(vec1, vec2) / (norm(vec1) * norm(vec2))

            sim_pos = np.dot(sent_vect, self.average_positive)/(np.linalg.norm(sent_vect)*np.linalg.norm(self.average_positive))

            sim_neut = np.dot(sent_vect, self.average_neutral)/(np.linalg.norm(sent_vect)*np.linalg.norm(self.average_neutral))

            sim_neg = np.dot(sent_vect, self.average_negative)/(np.linalg.norm(sent_vect)*np.linalg.norm(self.average_negative))

            return sim_pos[0], sim_neut[0], sim_neg[0]

        # if we were unable to extract any words from the given sentence, then we say the similarity is 0

        return 0, 0, 0

        
# creates and computs the average vectors for each sentiment from the data

mev = MeanEmbeddingVectorizer(w2v, pos_words, neg_words, neut_words)

mev = mev.fit(X_train['clean_selected_text'], X_train['sentiment'])
def calc_selected_text(df_row):

    

    words_in_tweet = df_row['text'].split()

    sentiment = df_row['sentiment']

    

    # we just return the entire tweet if the sentiment is neutral or if there are less than 3 words

    # almost every neutral tweet has selected text that is the same as the tweet

    # and most short tweets ended up using all words. 

    # This second part mostly saves computation time, but does increase accuracy a very small amount 

    # as the jaccard score is about 0.77 on average when just returning the tweet for short tweets

    if sentiment == 'neutral' or len(words_in_tweet) < 3:

        return df_row['text']

    

    # we get all of the possible subsets and sort them by length

    word_subsets = [words_in_tweet[i:j+1]

                    for i in range(len(words_in_tweet)) for j in range(i, len(words_in_tweet))]



    sorted_subsets = sorted(word_subsets, key=len)



    max_val = -10000000;

    final_subset = []



    # for each subset, we get the cosine similarity between that subset and the average vector for that sentiment

    # whichever one has the most similarity is the one that we return

    for subset in sorted_subsets:

        # clean the text, then split it on spaces to get an array

        cleaned_text = clean_text(' '.join(subset)).split(" ")

        

        # we get the cosine similarity between the subset and each average vector

        pos, neut, neg = mev.get_sent_dist(cleaned_text, sentiment)

#         print(pos, neut, neg)

        # then depending on which sentiment that tweet was, we figure which subset has the highest similarity

        val_to_check = pos if sentiment == 'positive' else neg

        if val_to_check > max_val:

            max_val = val_to_check

            final_subset = subset



    # then we just return the final_subset as a string

    # note we return the un-cleaned text as thats what the problem requires

    return " ".join(final_subset)
def calc_jaccard_df(data):

    # create/reset the columns we are going to be working with

    data['predicted_selection'] = ''

    data['jaccard'] = 0.0

    

    # for each sample in our data, we calculate the selected text and set the predicted_selection in our dataframe

    for index, row in data.iterrows():

        selected_text = calc_selected_text(row)

        data.loc[data['textID'] == row['textID'], ['predicted_selection']] = selected_text



    # calculate the jaccard score over the entire dataframe based off of the ground truth and our prediction

    data['jaccard'] = data.apply(lambda x: jaccard(x['selected_text'], x['predicted_selection']), axis = 1)

    # average all of the jaccard scores to get the total score for the validation set

    print('The jaccard score for the validation set is:', np.mean(data['jaccard']))

    

calc_jaccard_df(X_val)
# loop through the test dataset and calculate a prediction for each sample, then write it back into the dataframe

for index, row in test.iterrows():

    selected_text = calc_selected_text(row)

    sample.loc[sample['textID'] == row['textID'], ['selected_text']] = selected_text
# write the sample dataframe to a submissions file

sample.to_csv('submission.csv', index = False)