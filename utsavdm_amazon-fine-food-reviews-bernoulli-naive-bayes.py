# importing libraries:

import sqlite3                      # to save/ load the .sqlite files and perform SQL operations

import pandas as pd                 # dataframe ops

import numpy as np                  # array ops

from IPython.display import display # to view dataframe in a tabular format



# creating the connect object to connect with the database:

con = sqlite3.connect('../input/amazon-fine-food-reviews/database.sqlite')



# getting the values that are positive or negative and avoiding the ambiguous/ neutral (Score=3) reviews:

filtered_data = pd.read_sql_query("""

SELECT

    *

FROM 

    Reviews

WHERE 

    Score <> 3;

""", con)



# change Score from numbers to ratings as follows:

# if Score > 3 then 'Positive' rating i.e. 1

# if Score < 3 then 'Negative' rating i.e. 0, thus, eliminating Score = 3 cases:

def rate(x):

    if x < 3:

        return 0

    return 1



# replacing the numbers in the Score column with "Positive"/ "Negative" values:

positive_negative = filtered_data['Score']

positive_negative = positive_negative.map(rate)

filtered_data['Score'] = positive_negative
# Cleaning the data: de-duplicating



sorted_data = filtered_data.sort_values('ProductId', axis=0, inplace=False, ascending=True)

final = sorted_data.drop_duplicates(['UserId', 'ProfileName', 'Time','Text'], inplace=False, keep='first')



# there's also a scenario where helpfulness numerator is greater than helpfulness denominator which doesn't make any sense.

# because HelpfulnessNumerator is no. of YES (helpful reviews)

# and HelpfulnessDenominator is [no. of YES + no. of NO (not helpful reviews)]

print("Removing the below rows:\n", pd.read_sql_query("""

SELECT 

    *

FROM 

    Reviews

WHERE 

    HelpfulnessNumerator > HelpfulnessDenominator;

""", con))



# we thus only keep the rows where helpfulness numerator is less than or equal to the helpfulness denominator:

final = final[final.HelpfulnessNumerator <= final.HelpfulnessDenominator]



# resetting the index because many of the rows are deleted and their corresponding indices are missing:

final = final.reset_index(drop=True)

display(final)  # display the table in tabular format



# printing shape of the filtered/ modified data:

print("Shape of the dataframe: ", final.shape)



# can convert to interactive table using beakerx but not recommended as it takes long time since the dataset is huge:

#table = TableDisplay(final)

#print(table)

con.close()
# import statements:

import re   # to search for html tags, punctuations & special characters

import tqdm

import time

from tqdm import notebook



# importing gensim.models to implement Word2Vec:

import gensim

from gensim import models

from gensim.models import Word2Vec

from gensim.models import KeyedVectors



# Remove HTML tags - getting all the HTML tags and replacing them with blank spaces:

def cleanhtml(sentence):

    clean_text = re.sub('<.*?>', ' ', sentence)

    return clean_text



# Remove punctuations & special characters - getting all the punctuations and replacing them with blank spaces:

def cleanpunc(sentence):

    clean_text = re.sub(r'[@#$%\^&\*+=]', r'', sentence) # removing special characters

    clean_text = re.sub(r'[,.;\'"\-\!?:\\/|\[\]{}()]', r' ', clean_text) # removing punctuations

    return clean_text



final_clean_sentences = []



for sentence in notebook.tqdm(final['Text'].values):

    sentence = cleanhtml(sentence)

    sentence = cleanpunc(sentence)

    clean_sentence = []

    

    # for each word in the sentence, if it is alphabetic, we append it to the new list

    for word in sentence.split():

        if word.isalpha():

            clean_sentence.append(word.lower())

    

    # for each review in the 'Text' column, we create a list of words that appear in that sentence and store it in another list. 

    # basically, a list of lists - because that's how the model takes the input while training:

    final_clean_sentences.append(" ".join(clean_sentence))

    

print("Sentence cleaning completed!")

#print(final_clean_sentences[40:50])
from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



x = final_clean_sentences

y = final['Score']



count_vect = CountVectorizer(stop_words='english', ngram_range=(1,3), min_df=5)

count_vect = count_vect.fit_transform(x)

print("Shape of count_vect: ", count_vect.shape)



# Splitting into training & testing set:

x_train, x_test, y_train, y_test = train_test_split(count_vect, y, test_size=0.2, random_state=0)



# Using 10-fold cross validation to get the average generalization accuracy:

nb_classifier = BernoulliNB()

nb_score = cross_val_score(BernoulliNB(), x_train, y_train, cv=10, scoring='accuracy').mean()

print("Using Bernoulli Naive Bayes (Count Vectorizer) can give the generalization accuracy of: ~", nb_score)
# Implementing Naive Bayes:

nb_classifier = BernoulliNB()

nb_classifier = nb_classifier.fit(x_train, y_train)

nb_preds = nb_classifier.predict(x_test)



# Checking with confusion matrix:

cm = confusion_matrix(y_test, nb_preds)

import seaborn as sns

sns.heatmap(data=cm, annot=True)
TP = cm[0][0]

FP = cm[0][1]

FN = cm[1][0]

TN = cm[1][1]



print("True positives: ", TP)

print("True negatives: ", TN)

print("False positives: ", FP)

print("False negatives: ", FN)



acc = (TP+TN)/(TP+TN+FP+FN)

precision = TP/ (TP+FP)

recall = TP/ (TP+FN)

f1_score = 2*precision*recall/ (precision + recall)



print("========= Final performance evaluation =========")

print("Accuracy: ", acc)

print("Precision: ", precision)

print("Recall: ", recall)

print("F-1 Score: ", f1_score)

print("==============================================")
x = final_clean_sentences

y = final['Score']



vect = TfidfVectorizer(stop_words='english', ngram_range=(1,3), min_df=5, sublinear_tf=True)

tfidf_vect = vect.fit_transform(x)

print("Shape of tfidf_vect: ", tfidf_vect.shape)



# Splitting into training & testing set:

x_train, x_test, y_train, y_test = train_test_split(tfidf_vect, y, test_size=0.2, random_state=0)



# Using 10-fold cross validation to get the average generalization accuracy:

nb_classifier = BernoulliNB()

nb_score = cross_val_score(BernoulliNB(), x_train, y_train, cv=10, scoring='accuracy').mean()

print("Using Bernoulli Naive Bayes (TF-IDF Vectorizer) can give the generalization accuracy of: ~", nb_score)
# Implementing Naive Bayes:

nb_classifier = BernoulliNB()

nb_classifier = nb_classifier.fit(x_train, y_train)

nb_preds = nb_classifier.predict(x_test)



# Checking with confusion matrix:

cm = confusion_matrix(y_test, nb_preds)

import seaborn as sns

sns.heatmap(data=cm, annot=True)
TP = cm[0][0]

FP = cm[0][1]

FN = cm[1][0]

TN = cm[1][1]



print("True positives: ", TP)

print("True negatives: ", TN)

print("False positives: ", FP)

print("False negatives: ", FN)



acc = (TP+TN)/(TP+TN+FP+FN)

precision = TP/ (TP+FP)

recall = TP/ (TP+FN)

f1_score = 2*precision*recall/ (precision + recall)



print("========= Final performance evaluation =========")

print("Accuracy: ", acc)

print("Precision: ", precision)

print("Recall: ", recall)

print("F-1 Score: ", f1_score)

print("==============================================")