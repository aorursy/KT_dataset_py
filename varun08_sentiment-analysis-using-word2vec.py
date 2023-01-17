# Firstly, please note that the performance of google word2vec is better on big datasets. 

# In this example we are considering only 25000 training examples from the imdb dataset.

# Therefore, the performance is similar to the "bag of words" model.



# Importing libraries

import numpy as np

import pandas as pd

# BeautifulSoup is used to remove html tags from the text

from bs4 import BeautifulSoup 

import re # For regular expressions



# Stopwords can be useful to undersand the semantics of the sentence.

# Therefore stopwords are not removed while creating the word2vec model.

# But they will be removed  while averaging feature vectors.

from nltk.corpus import stopwords



# Read data from files

train = pd.read_csv("../input/labeledTrainData.tsv", header=0,\

                    delimiter="\t", quoting=3)



test = pd.read_csv("../input/testData.tsv",header=0,\

                    delimiter="\t", quoting=3)
# This function converts a text to a sequence of words.

def review_wordlist(review, remove_stopwords=False):

    # 1. Removing html tags

    review_text = BeautifulSoup(review).get_text()

    # 2. Removing non-letter.

    review_text = re.sub("[^a-zA-Z]"," ",review_text)

    # 3. Converting to lower case and splitting

    words = review_text.lower().split()

    # 4. Optionally remove stopwords

    if remove_stopwords:

        stops = set(stopwords.words("english"))     

        words = [w for w in words if not w in stops]

    

    return(words)


# word2vec expects a list of lists.

# Using punkt tokenizer for better splitting of a paragraph into sentences.



import nltk.data

#nltk.download('popular')



tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# This function splits a review into sentences

def review_sentences(review, tokenizer, remove_stopwords=False):

    # 1. Using nltk tokenizer

    raw_sentences = tokenizer.tokenize(review.strip())

    sentences = []

    # 2. Loop for each sentence

    for raw_sentence in raw_sentences:

        if len(raw_sentence)>0:

            sentences.append(review_wordlist(raw_sentence,\

                                            remove_stopwords))



    # This returns the list of lists

    return sentences



sentences = []

print("Parsing sentences from training set")

for review in train["review"]:

    sentences += review_sentences(review, tokenizer)

    
# Importing the built-in logging module

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Creating the model and setting values for the various parameters

num_features = 300  # Word vector dimensionality

min_word_count = 40 # Minimum word count

num_workers = 4     # Number of parallel threads

context = 10        # Context window size

downsampling = 1e-3 # (0.001) Downsample setting for frequent words



# Initializing the train model

from gensim.models import word2vec

print("Training model....")

model = word2vec.Word2Vec(sentences,\

                          workers=num_workers,\

                          size=num_features,\

                          min_count=min_word_count,\

                          window=context,

                          sample=downsampling)



# To make the model memory efficient

model.init_sims(replace=True)



# Saving the model for later use. Can be loaded using Word2Vec.load()

model_name = "300features_40minwords_10context"

model.save(model_name)

# Few tests: This will print the odd word among them 

model.wv.doesnt_match("man woman dog child kitchen".split())



model.wv.doesnt_match("france england germany berlin".split())

# This will print the most similar words present in the model

model.wv.most_similar("man")

model.wv.most_similar("awful")

# This will give the total number of words in the vocabolary created from this dataset

model.wv.syn0.shape
# Function to average all word vectors in a paragraph

def featureVecMethod(words, model, num_features):

    # Pre-initialising empty numpy array for speed

    featureVec = np.zeros(num_features,dtype="float32")

    nwords = 0

    

    #Converting Index2Word which is a list to a set for better speed in the execution.

    index2word_set = set(model.wv.index2word)

    

    for word in  words:

        if word in index2word_set:

            nwords = nwords + 1

            featureVec = np.add(featureVec,model[word])

    

    # Dividing the result by number of words to get average

    featureVec = np.divide(featureVec, nwords)

    return featureVec



# Function for calculating the average feature vector

def getAvgFeatureVecs(reviews, model, num_features):

    counter = 0

    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")

    for review in reviews:

        # Printing a status message every 1000th review

        if counter%1000 == 0:

            print("Review %d of %d"%(counter,len(reviews)))

            

        reviewFeatureVecs[counter] = featureVecMethod(review, model, num_features)

        counter = counter+1

        

    return reviewFeatureVecs

# Calculating average feature vector for training set

clean_train_reviews = []

for review in train['review']:

    clean_train_reviews.append(review_wordlist(review, remove_stopwords=True))

    

trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)
# Calculating average feature vactors for test set     

clean_test_reviews = []

for review in test["review"]:

    clean_test_reviews.append(review_wordlist(review,remove_stopwords=True))

    

testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)
# Fitting a random forest classifier to the training data

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100)

    

print("Fitting random forest to training data....")    

forest = forest.fit(trainDataVecs, train["sentiment"])

    
# Predicting the sentiment values for test data and saving the results in a csv file 

result = forest.predict(testDataVecs)

output = pd.DataFrame(data={"id":test["id"], "sentiment":result})

output.to_csv( "output.csv", index=False, quoting=3 )