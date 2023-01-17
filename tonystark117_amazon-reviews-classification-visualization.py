# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import sqlite3

import re

from bs4 import BeautifulSoup

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer,PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer

from tqdm import tqdm



from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from gensim.models import Word2Vec,KeyedVectors



from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.metrics import precision_score,recall_score,roc_curve,roc_auc_score

from sklearn.manifold import TSNE

from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Connection = sqlite3.connect('/kaggle/input/amazon-fine-food-reviews/database.sqlite')

Amazon_Reviews = pd.read_sql_query("""select * from Reviews where Score!=3 LIMIT 10000""",Connection)

Amazon_Reviews['Score'] = Amazon_Reviews['Score'].apply(lambda x: 1 if x>3 else 0)    
Amazon_Reviews.describe
pd.read_sql_query("""

SELECT UserId, ProductId, ProfileName, Time, Score, Text, COUNT(*)

FROM Reviews

GROUP BY UserId

HAVING COUNT(*)>1

""", Connection).head()
pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score != 3 AND UserId="AR5J8UI46CURR"

ORDER BY ProductID

""", Connection).head()
#Sorting data according to ProductId in ascending order

Amazon_Reviews_Sorted=Amazon_Reviews.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
#Deduplication of entries

print("Shape of Amazon Reviews Before Removing duplicates is",Amazon_Reviews_Sorted.shape)

Amazon_Reviews_Sorted_Filtered_Duplicates=Amazon_Reviews_Sorted.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)

print("Shape of Amazon Reviews After Removing duplicates is",Amazon_Reviews_Sorted_Filtered_Duplicates.shape)
print("{}% of 100 Retained after removing duplicates".format((Amazon_Reviews_Sorted_Filtered_Duplicates.shape[0]/Amazon_Reviews_Sorted.shape[0])*100))
Amazon_Reviews_Dataset = Amazon_Reviews_Sorted_Filtered_Duplicates[Amazon_Reviews_Sorted_Filtered_Duplicates['HelpfulnessNumerator']<=Amazon_Reviews_Sorted_Filtered_Duplicates['HelpfulnessDenominator']]

print("{}% of 100 Retained after Cleaning".format((Amazon_Reviews_Dataset.shape[0]/Amazon_Reviews_Sorted_Filtered_Duplicates.shape[0])*100))

print(Amazon_Reviews_Dataset['Score'].value_counts())
def Deconstracted(Phrase):

    #specific

    Phrase=re.sub(r"can\'t","can not",Phrase)

    Phrase=re.sub(r"won\'t","will not",Phrase)

    #general

    Phrase=re.sub(r"\'re","are",Phrase)

    Phrase=re.sub(r"\'t","not",Phrase)

    Phrase=re.sub(r"\'s","is",Phrase)

    Phrase=re.sub(r"\'ll","will",Phrase)

    Phrase=re.sub(r"\'ve","have",Phrase)

    Phrase=re.sub(r"\'d","would",Phrase)

    return Phrase
#Custom_StopWords=set(word for word in stopwords.words("english") if word not in ["no","not"])

Custom_StopWords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\

            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \

            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\

            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \

            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \

            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \

            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\

            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\

            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\

            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \

            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \

            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\

            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\

            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \

            'won', "won't", 'wouldn', "wouldn't"])
def Preprocessing(reviews):

    Preprocessed_Reviews=[]

    for document in tqdm(reviews['Text'].values):

        document=re.sub(r"http\S+","",document)

        document=BeautifulSoup(document,'lxml').get_text()

        document=Deconstracted(document)

        document=re.sub(r"\S*\d\S*","",document).strip()

        document=re.sub('[^A-Za-z]+'," ",document)

        document=' '.join(word.lower() for word in document.split() if word.lower() not in Custom_StopWords)

        Preprocessed_Reviews.append(document)

    return Preprocessed_Reviews
Reviews=Preprocessing(Amazon_Reviews_Dataset)
cnt_vector=CountVectorizer()

Reviews_Count_Vector_Features=cnt_vector.fit_transform(Reviews)

print(Reviews_Count_Vector_Features.toarray().shape)
cnt_vector_ngram=CountVectorizer(ngram_range=(1,2),min_df=10)

Reviews_Count_Vector_ngram_Features=cnt_vector_ngram.fit_transform(Reviews)

print(Reviews_Count_Vector_ngram_Features.toarray().shape)
tfidf=TfidfVectorizer(ngram_range=(1,2),min_df=10)

tfidf_features=tfidf.fit_transform(Reviews)

print(tfidf_features.toarray().shape)
List_Of_Sentences=[]

for sentence in Reviews:

    List_Of_Sentences.append(sentence.split())
word2vec_model=Word2Vec(List_Of_Sentences,size=50,workers=7,min_count=5)

word2vec_model_vocabulary = list(word2vec_model.wv.vocab)
AVG_WORD2VEC_Sentence_Vectors=[]

for sentence in tqdm(List_Of_Sentences):

    sentence_vector=np.zeros(50)

    count_vector=0

    for word in sentence:

        if word in word2vec_model_vocabulary:

            vector = word2vec_model.wv[word]

            sentence_vector+=vector

            count_vector+=1

    if count_vector!=0:

        sentence_vector/=count_vector

    AVG_WORD2VEC_Sentence_Vectors.append(sentence_vector)
tfidf_Word2Vec=TfidfVectorizer()

tfidf_features=tfidf_Word2Vec.fit_transform(Reviews)



Dictionary =dict(zip(tfidf_Word2Vec.get_feature_names(),list(tfidf_Word2Vec.idf_)))

TFIDF_Features_Words=tfidf_Word2Vec.get_feature_names()
TFIDF_WORD2VEC_Sentence_Vectors=[]

for sentence in tqdm(List_Of_Sentences):

    TFIDF_Sentence_Vector=np.zeros(50)

    Weight_Vector=0

    for word in sentence:

        if word in word2vec_model_vocabulary and word in TFIDF_Features_Words:

            word_vector=word2vec_model.wv[word]

            tfidf=Dictionary[word]*(sentence.count(word)/len(sentence))

            TFIDF_Sentence_Vector+=word_vector*tfidf

            Weight_Vector+=tfidf

    if Weight_Vector!=0:

        TFIDF_Sentence_Vector/=Weight_Vector

    TFIDF_WORD2VEC_Sentence_Vectors.append(TFIDF_Sentence_Vector)
import time

plt.figure(figsize=(12,10))

cnt_vector_TSNE = TSNE(n_components=2,perplexity=50, learning_rate=200)

start_time=time.time()

X_Cnt_Vector_TSNE=cnt_vector_TSNE.fit_transform(Reviews_Count_Vector_Features.toarray())

Data = np.hstack((X_Cnt_Vector_TSNE,Amazon_Reviews_Dataset['Score'].values.reshape(-1,1)))

TSNE_DF=pd.DataFrame(Data,columns=['Dimension_x','Dimension_y','Score'])

colors = {0:'red', 1:'blue', 2:'green'}

plt.scatter(TSNE_DF['Dimension_x'], TSNE_DF['Dimension_y'], c=TSNE_DF['Score'].apply(lambda x: colors[x]))

plt.show()

stop_time=time.time()

Total_Time=(stop_time-start_time)/60

print(Total_Time)
import time

plt.figure(figsize=(12,10))

cnt_vector_TSNE = TSNE(n_components=2,perplexity=40, learning_rate=200)

start_time=time.time()

X_Cnt_Vector_TSNE=cnt_vector_TSNE.fit_transform(Reviews_Count_Vector_Features.toarray())

Data = np.hstack((X_Cnt_Vector_TSNE,Amazon_Reviews_Dataset['Score'].values.reshape(-1,1)))

TSNE_DF=pd.DataFrame(Data,columns=['Dimension_x','Dimension_y','Score'])

colors = {0:'red', 1:'blue', 2:'green'}

plt.scatter(TSNE_DF['Dimension_x'], TSNE_DF['Dimension_y'], c=TSNE_DF['Score'].apply(lambda x: colors[x]))

plt.show()

stop_time=time.time()

Total_Time=(stop_time-start_time)/60

print(Total_Time)