import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from bs4 import BeautifulSoup
import nltk

import os

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/labeledTrainData.tsv", header = 0, delimiter = '\t')
# import trainning data
from nltk.corpus import stopwords

def review_to_words( raw_review):
        review_text = BeautifulSoup(raw_review, features="html5lib").get_text()         # remove HTML
        letters_only = re.sub("[^a-zA-Z]", " ", review_text)            # Remove non-letters
        words = letters_only.lower().split()                            # to lower case, split to individual words
        stops = set(stopwords.words("english"))                         # in python, searching a set is much faster than searching a list, so convert the stop words to a set
        meaningful_words = [w for w in words if not w in stops]         # remove stop words
        return(" ".join( meaningful_words ))                            # Join the words back into one string separated by space, and return the result.

num_reviews = train["review"].size
clean_train_reviews = []

for i in range( 0, num_reviews):
        clean_train_reviews.append( review_to_words( train["review"][i]))
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["sentiment"] )
# Read the test data
test = pd.read_csv("../input/testData.tsv", header=0, delimiter="\t", \
                   quoting=3 )

# Verify that there are 25,000 rows and 2 columns


# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = [] 


for i in range(0,num_reviews):
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "submit.csv", index=False, quoting=3 )