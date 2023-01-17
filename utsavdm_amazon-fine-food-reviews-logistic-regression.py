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

# if Score > 3 then 'Positive' rating

# if Score < 3 then 'Negative' rating, thus, eliminating Score = 3 cases:

def rate(x):

    if x < 3:

        return "Negative"

    return "Positive"



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
# importing gensim.models to implement Word2Vec:

import gensim

from gensim import models

from gensim.models import Word2Vec

from gensim.models import KeyedVectors



# in the filename, 300 stands for 300 dimensions:

# loading the model, this is a very high resource consuming task: 

g_trained_model = KeyedVectors.load_word2vec_format('../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin', binary=True)

print("Model loaded successfully: ", type(g_trained_model))
# Just testing the model:

print('Most similar to the word "pizza"', g_trained_model.most_similar('pizza'))
# import statements:

import re   # to search for html tags, punctuations & special characters



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



for sentence in final['Text'].values:

    sentence = cleanhtml(sentence)

    sentence = cleanpunc(sentence)

    clean_sentence = []

    

    # for each word in the sentence, if it is alphabetic, we append it to the new list

    for word in sentence.split():

        if word.isalpha():

            clean_sentence.append(word.lower())

    

    # for each review in the 'Text' column, we create a list of words that appear in that sentence and store it in another list. 

    # basically, a list of lists - because that's how the model takes the input while training:

    final_clean_sentences.append(clean_sentence)

    

print("Sentence cleaning completed!")

del clean_sentence, sorted_data, filtered_data, positive_negative
# below imports are used only to show the progress bar:

import tqdm

import time



vectored_sentences = []

not_converted = set()  # used to keep track of how many words are not converted to vector

print("Average Word2Vec calculations started...")



# its ok to ignore tqdm_notebook in the for loop below, just kept it for ETA:

for sentence in final_clean_sentences:

    vec_sent = np.zeros(300)        # since the size of g_trained_model is 300

    for word in sentence:

        if word in g_trained_model:

            vectored_word = g_trained_model.wv[word]

            vec_sent += vectored_word

        else:

             not_converted.add(word)

                

    vec_sent/=len(sentence)

    vectored_sentences.append(vec_sent)



print("\nAvg. Word2Vec calculations completed!")

print("First element of Avg. Word2Vec vectored sentences:\n", vectored_sentences[0])

print("-"*60)

del g_trained_model
# Separating 'X' & 'y':

score = list(final['Score'])

x = pd.DataFrame(vectored_sentences)

print("X shape: ", x.shape)

y = pd.DataFrame(score, columns=['rating'])

y.loc[y['rating']=='Positive', 'rating'] = 1

y.loc[y['rating']=='Negative', 'rating'] = 0

print(y.shape)

print(y.dtypes)



# Here y has the dtypes as 'object' which can be used with fit() method of any ML algorithm and thus need to convert it to int object

y=y.astype('int')

print(y.dtypes)



# Standardization:

from sklearn.preprocessing import StandardScaler

x_scaler = StandardScaler()

x = x_scaler.fit_transform(x)

print(x)

#print(x_scaler.transform(y))



# Imputation/ Handling the missing values:

from sklearn.impute import SimpleImputer

x_impute = SimpleImputer(missing_values=np.nan, strategy='mean')

x = x_impute.fit_transform(x)

print(x)



# Creating training & testing datasets:

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)

print("x_train", x_train.shape)

print("x_test", x_test.shape)

print("y_train", y_train.shape)

print("y_test", y_test.shape)

del final, x_scaler, x_impute
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression



# Testing the model's performance using 10-fold cross-validation:

LR = LogisticRegression(penalty='l1', random_state=0, solver='saga', n_jobs=-1)

lr_score = cross_val_score(LR, x_train, y_train.values.ravel(), cv=10, scoring='accuracy').mean()

print("Using Logistic Regression, can give the generalization accuracy of: ~", lr_score)
# Implementing Logistic Regression:

LR = LogisticRegression(penalty='l2', random_state=0, solver='saga', n_jobs=-1)

LR = LR.fit(x_train, y_train.values.ravel())

lr_preds = LR.predict(x_test)
# Checking with confusion matrix:

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, lr_preds)

import seaborn as sns

print("Confusion matrix:")

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



print("\n=========Final performance evaluation=========")

print("Accuracy: ", acc)

print("Precision: ", precision)

print("Recall: ", recall)

print("F-1 Score: ", f1_score)

print("==============================================")