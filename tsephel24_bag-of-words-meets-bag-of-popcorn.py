import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
    
import sys
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import time
import seaborn as sns
from pandas_datareader import wb, data as web
import os

%matplotlib inline

train = pd.read_csv("../input/labeledTrainData.tsv", header=0, \
                    delimiter="\t")
train.head()
import re
from nltk.corpus import stopwords 

from bs4 import BeautifulSoup
num_reviews = train["review"].size
clean_test_reviews = []
for i in range( 0, num_reviews ):
    clean_test_reviews.append( review_to_words( train["review"][i] ) )
from bs4 import BeautifulSoup
print ("Cleaning and parsing the training set movie reviews...\n")
clean_train_reviews = []
for i in range( 0, num_reviews ):
    if( (i+1)%1000 == 0 ):
        clean_train_reviews.append( review_to_words( train["review"][i] ))
num_reviews = len(test["review"])
clean_test_reviews = [] 
from bs4 import BeautifulSoup
print ("Cleaning and parsing the test set movie reviews...\n")
for i in range(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print ("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words( test["review"][i] )
    clean_train_reviews.append( clean_review )


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()
print ("Training the random forest...")
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100) 

forest = forest.fit( train_data_features, train["sentiment"] )


result = forest.predict(train_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "submit.csv", index=False, quoting=3 )