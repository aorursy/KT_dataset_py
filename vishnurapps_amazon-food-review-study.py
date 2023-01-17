import numpy as np

import pandas as pd

import sqlite3

import string

import matplotlib.pyplot as plt

import seaborn as sn
#Using sqlite3 to read the data

con = sqlite3.connect('../input/database.sqlite')
#Filtering positive(5 & 4 stars) and negative(1 & 2 stars) reviews and discarding 3 star reviews.

filtered_data = pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score != 3

""", con)
#Give reviews greater than 3 positive and less than 3 as negative

def partition(x):

    if x < 3:

        return 'negative'

    return 'positive'
#Changing the reviews based on the star value

actualScore = filtered_data["Score"]

positiveNegative = actualScore.map(partition)

filtered_data['Score'] = positiveNegative
filtered_data.shape
filtered_data.head()
display = pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score != 3 AND UserId = "AR5J8UI46CURR"

ORDER BY ProductID

""", con)

display
#Sorting data according to ProductId in ascending order

sorted_data = filtered_data.sort_values('ProductId', axis=0, ascending=True)
#Deduplication of entries

final = sorted_data.drop_duplicates(subset={"UserId", "ProfileName", "Time", "Text"}, keep="first", inplace=False)
final.shape #If we comare this with the size of input data, we can see that there is a significat reduction. Size of the read data was final.shape #If we comare this with the size of input data, we can see that there is a significat reduction. Size of the read data was (525814, 10) 
#Lets check how much data is left after removing duplicates

(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100
#In reviews the helpful numerator has to be greater that helpful denominator. But there are some reviews that has a probelm with this

display = pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score != 3 AND Id = 44737 OR Id = 64422

ORDER BY ProductID

""", con)

display
final = final[final.HelpfulnessNumerator <= final.HelpfulnessDenominator]
#We have this many reviews left

final.shape
#Number of positive and negative reviews in dataset

final['Score'].value_counts()
import scipy

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
final_counts = count_vect.fit_transform(final['Text'].values)
type(final_counts) #Here the final count is a sparse matrix
final_counts.get_shape()
import re

import string

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem import SnowballStemmer

from nltk.stem.wordnet import WordNetLemmatizer



from gensim.models import Word2Vec

from gensim.models import KeyedVectors

import pickle



from tqdm import tqdm

import os
stop = set(stopwords.words('english'))

sno = SnowballStemmer('english')
print(stop)
print(sno.stem('tasty'))
print(sno.stem('worked'))
def cleanHtml(sentence):

    cleanr = re.compile('<.?>')

    cleartext = re.sub(cleanr, ' ', sentence)

    return cleartext

def cleanPunc(sentence):

    cleaned = re.sub(r'[?|!|\'|\"|#]', r'',sentence)

    cleaned = re.sub(r'[.|,|)|(|\|/]', r'',cleaned)

    return cleaned
#removing html and punctuations

i = 0

str1 = ' '

final_string = []

all_positive_words=[] #store positive reviews

all_negative_words=[] #store negative reviews

s = ''



for sent in final['Text'].values:

    filtered_sentence=[]

    sent = cleanHtml(sent)

    for w in sent.split():

        for cleaned_word in cleanPunc(w).split():

            if((cleaned_word.isalpha()) & (len(cleaned_word)>2)):

                if(cleaned_word.lower() not in stop):

                    s=(sno.stem(cleaned_word.lower())).encode('utf8')

                    filtered_sentence.append(s)

                    if(final['Score'].values)[i] == 'positive':

                        all_positive_words.append(s)

                    if(final['Score'].values)[i] == 'positive':

                        all_negative_words.append(s)

                else:

                    continue

            else:

                continue

    str1=b" ".join(filtered_sentence)

    final_string.append(str1)

    i+=1

final['CleanedText'] = final_string
final.head(3)
conn = sqlite3.connect('final.sqlite')

c=conn.cursor()

conn.text_factory = str

final.to_sql('Reviews', conn, schema=None, if_exists='replace')