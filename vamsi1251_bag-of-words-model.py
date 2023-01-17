import numpy as np 
import pandas as pd 
import re
from bs4 import BeautifulSoup
import nltk

import os
train = pd.read_csv("../input/labeledTrainData.tsv", header = 0, delimiter = '\t')
from nltk.corpus import stopwords
def review_to_words( raw_review):
        review_text = BeautifulSoup(raw_review, features="html5lib").get_text()
        letters_only = re.sub("[^a-zA-Z]", " ", review_text)            
        words = letters_only.lower().split() 
        stops = set(stopwords.words("english"))                         
        meaningful_words = [w for w in words if not w in stops]         
        return(" ".join( meaningful_words ))
num_reviews = train["review"].size
clean_train_reviews = []

for i in range( 0, num_reviews):
        clean_train_reviews.append( review_to_words( train["review"][i]))
from sklearn.feature_extraction.text import CountVectorizer 
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 150) 
forest = forest.fit( train_data_features, train["sentiment"] )
test = pd.read_csv("../input/testData.tsv", header=0, delimiter="\t", \
                   quoting=3 )

num_reviews = len(test["review"])
clean_test_reviews = [] 


for i in range(0,num_reviews):
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )
    
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

output.to_csv( "submit.csv", index=False, quoting=3 )
