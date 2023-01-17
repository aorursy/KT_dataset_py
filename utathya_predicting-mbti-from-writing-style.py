# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
################################################# import libraries ###########################################

import pandas as pd
import os
from nltk.corpus import stopwords
import string
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import operator
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import time
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering

def rem_sw(df):
    # Downloading stop words
    stop_words = set(stopwords.words('english'))
    stop_words |= set(['infj', 'intp', 'infp', 'enfp', 'intj', 'entp', 'istp', 'entj', 'isfp', 'enfj', 'isfj', 'istj', 'estp', 'esfp', 'estj', 'esfj'])

    print(type(df))
    count = 0
    for sentence in df:
        sentence = [word for word in sentence.lower().split() if word not in stop_words]
        sentence = ' '.join(sentence)
        df.loc[count] = sentence
        count+=1
    return(df)

def rem_punc(df):
    count = 0
    for s in df:
        cleanr = re.compile('<.*?>')
        s = re.sub(r'\d+', '', s)
        s = re.sub(cleanr, '', s)
        s = re.sub("'", '', s)
        s = re.sub(r'\W+', ' ', s)
        s = s.replace('_', '')
        df.loc[count] = s
        count+=1
    return(df)

def lemma(df):

    lmtzr = WordNetLemmatizer()

    count = 0
    stemmed = []
    for sentence in df:    
        word_tokens = word_tokenize(sentence)
        for word in word_tokens:
            stemmed.append(lmtzr.lemmatize(word))
        sentence = ' '.join(stemmed)
        df.iloc[count] = sentence
        count+=1
        stemmed = []
    return(df)

def stemma(df):

    stemmer = SnowballStemmer("english") #SnowballStemmer("english", ignore_stopwords=True)

    count = 0
    stemmed = []
    for sentence in df:
        word_tokens = word_tokenize(sentence)
        for word in word_tokens:
            stemmed.append(stemmer.stem(word))
        sentence = ' '.join(stemmed)
        df.iloc[count] = sentence
        count+=1
        stemmed = []
    return(df)

def get_feature(df, number):
    
    feature_list = []
    # create an instance for tree feature selection
    tree_clf = ExtraTreesClassifier()

    # first create arrays holding input and output data

    # Vectorizing Train set
    cv = CountVectorizer(analyzer='word')
    x_train = cv.fit_transform(df['posts'])

    # Creating an object for Label Encoder and fitting on target strings
    le = LabelEncoder()
    y = le.fit_transform(df['type'])

    # fit the model
    tree_clf.fit(x_train, y)
    
    # Preparing variables
    importances = tree_clf.feature_importances_
    feature_names = cv.get_feature_names()
    feature_imp_dict = dict(zip(feature_names, importances))
    sorted_features = sorted(feature_imp_dict.items(), key=operator.itemgetter(1), reverse=True)
    indices = np.argsort(importances)[::-1]

    # Create the feature list
    for f in range(number):
        feature_list.append(sorted_features[f][0])
    
    return(feature_list)

def print_feature(df):
    
    # create an instance for tree feature selection
    tree_clf = ExtraTreesClassifier()

    # first create arrays holding input and output data

    # Vectorizing Train set
    cv = CountVectorizer(analyzer='word')
    x_train = cv.fit_transform(df['posts'])

    # Creating an object for Label Encoder and fitting on target strings
    le = LabelEncoder()
    y = le.fit_transform(df['type'])

    # fit the model
    tree_clf.fit(x_train, y)

    # Preparing variables
    importances = tree_clf.feature_importances_
    feature_names = cv.get_feature_names()
    feature_imp_dict = dict(zip(feature_names, importances))
    sorted_features = sorted(feature_imp_dict.items(), key=operator.itemgetter(1), reverse=True)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(20):
        print("feature %d : %s (%f)" % (indices[f], sorted_features[f][0], sorted_features[f][1]))

    # Plot the feature importances of the forest
    plt.figure(figsize = (20,20))
    plt.title("Feature importances")
    plt.bar(range(100), importances[indices[:100]],
           color="r", align="center")
    plt.xticks(range(100), sorted_features[:100], rotation=90)
    plt.xlim([-1, 100])
    plt.show()

    return()
# Create master dataset
file_path = os.path.join(os.path.abspath(''), '../input/mbti_clean_train.csv')
df_mbti = pd.read_csv(file_path, encoding='ISO-8859-1', engine='c', index_col=0)

df_mbti.drop

mbti_dict = {0: 'INFJ', 1: 'ENTP', 2: 'INTJ', 3: 'ENTJ', 4: 'ENFJ', 5: 'ENFP', 6: 'ISFP', 7: 'INTP', 8: 'ISTP', 9: 'ISFJ'
             , 10: 'ISTJ', 11: 'ESTP', 12: 'ESFP', 13: 'INFP', 14: 'ESTJ', 15: 'ESFJ'}

df_mbti_pred = df_mbti.replace(to_replace=mbti_dict.values(), value=mbti_dict.keys())
df_mbti_pred.loc[len(df_mbti_pred)+1, 'posts'] = input("A paragraph of your writing: ")

# Remove stop words
stop_words = set(stopwords.words('english'))
sentence = df_mbti_pred.iloc[-1, 1] 
sentence = [word for word in df_mbti_pred.iloc[-1, 1].lower().split() if word not in stop_words]
sentence = ' '.join(df_mbti_pred.iloc[-1, 1])
df_mbti_pred.iloc[-1, 1] = sentence

# Remove punctuations
s = df_mbti_pred.iloc[-1, 1]
cleanr = re.compile('<.*?>')
s = re.sub(r'\d+', '', s)
s = re.sub(cleanr, '', s)
s = re.sub("'", '', s)
s = re.sub(r'\W+', ' ', s)
s = s.replace('_', '')
df_mbti_pred.iloc[-1, 1] = s

# Lemmatising
lmtzr = WordNetLemmatizer()
stemmed = []
sentence = df_mbti_pred.iloc[-1, 1]
word_tokens = word_tokenize(sentence)
for word in word_tokens:
    stemmed.append(lmtzr.lemmatize(word))
sentence = ' '.join(stemmed)
df_mbti_pred.iloc[-1, 1] = sentence

#Stemmatisation
stemmer = SnowballStemmer("english") #SnowballStemmer("english", ignore_stopwords=True)
stemmed = []
sentence = df_mbti_pred.iloc[-1, 1]
word_tokens = word_tokenize(sentence)
for word in word_tokens:
    stemmed.append(stemmer.stem(word))
sentence = ' '.join(stemmed)
df_mbti_pred.iloc[-1, 1] = sentence

# Remove MBTI type words
mbti_words = ['infj', 'intp', 'infp', 'enfp', 'intj', 'entp', 'istp', 'entj', 'isfp', 'enfj', 'isfj', 'istj', 'estp', 'esfp', 'estj', 'esfj']
sentence = df_mbti_pred.iloc[-1, 1] 
sentence = [word for word in df_mbti_pred.iloc[-1, 1].lower().split() if word not in mbti_words]
sentence = ' '.join(df_mbti_pred.iloc[-1, 1])
df_mbti_pred.iloc[-1, 1] = sentence

# Data preparation
vect_algo = TfidfVectorizer(stop_words='english', analyzer='word')
X = vect_algo.fit_transform(df_mbti_pred.posts)

# Encoding target data
# Creating an object and fitting on target strings
y = df_mbti_pred.type[:-1]

########################################## Decision tree ##########################
# Create decision tree object
clf = DecisionTreeClassifier(criterion='entropy')

# fit the model
clf.fit(X[:-1], y)

# predict the outcome for testing data
predictions = clf.predict(X[-1])

print("Your MBTI writing style is", mbti_dict[int(predictions)])
############################################# Loading Data for Project ##########################################

# Create master dataset
file_path = os.path.join(os.path.abspath(''), '../input/mbti_1.csv')
df_mbti = pd.read_csv(file_path, encoding='ISO-8859-1', engine='c')
df_mbti.head(1)
df_mbti['posts'] = rem_sw(df_mbti['posts'])
df_mbti['posts'] = rem_punc(df_mbti['posts'])
df_mbti['posts'] = lemma(df_mbti['posts'])
df_mbti['posts'] = stemma(df_mbti['posts'])
df_mbti['posts'] = rem_sw(df_mbti['posts'])

df_mbti.to_csv("mbti_clean_train.csv")

print_feature(df_mbti)
########################## WordCloud All ##################################

# Creating a list of train and test data to analyse
mbti_list = df_mbti["posts"].unique().tolist()
mbti_bow = " ".join(mbti_list)

# Create a word cloud for psitive words
mbti_wordcloud = WordCloud().generate(mbti_bow)

# Show the created image of word cloud
plt.figure(figsize=(20, 20))
plt.imshow(mbti_wordcloud)
plt.show()
################### For split and check ####################
vect_algo = TfidfVectorizer(stop_words='english', analyzer='word')
X = vect_algo.fit_transform(df_mbti.posts)

# Encoding target data
# Creating an object and fitting on target strings
le = LabelEncoder()
y = le.fit_transform(df_mbti.type)

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=13, stratify=y)

Counter(y_train)
########################################## Decision tree ##########################

# Starting time for time calculations
start_time = time.time()

# Create decision tree object
clf = DecisionTreeClassifier(criterion='entropy')

# fit the model
clf.fit(X_train, y_train)

# predict the outcome for testing data
predictions = clf.predict(X_test)

# # check the accuracy of the model
accuracy = accuracy_score(y_test, predictions)

# Visualising performance: accuracy is adjusted in case model prediction is inverse in the binary state
if accuracy < 0.5:
    accuracy = 1 - accuracy
print("The time taken to execute is %s seconds" % (time.time() - start_time))
print("The accuracy of the model is %.2f%%" % (accuracy*100))