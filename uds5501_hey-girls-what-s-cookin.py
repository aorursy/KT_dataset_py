# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_path = '../input/train.json'
test_path = '../input/test.json'
df_train = pd.read_json(train_path)
df_test = pd.read_json(test_path)
len(df_train)
df_train.head()
unique_labels = list(df_train['cuisine'].unique())
plt.figure(figsize=(18,6))
sns.countplot(df_train['cuisine'])
%%time


ingredient_mapping = {}
for i in unique_labels:
    ingredient_mapping[i] = []

for i in range(len(df_train)):
    label = df_train.iloc[i][0]
    ingredient_list = df_train.iloc[i][-1]
    for ingredient in ingredient_list:
        if ingredient in ingredient_mapping[label] is True:
            continue
        else:
            ingredient_mapping[label].append(ingredient)
            
        
#ingredient_mapping['italian']
df_train.head()
#df[['add','sub']] = df['A'].apply([plus, minus])
print(unique_labels)
def getGreekScore(ingredient_array):
    count = 0
    for ingredient in ingredient_array:
        if ingredient in ingredient_mapping['greek']:
            count += 1
    return (count / float(len(ingredient_array)))

def getsouthern_usScore(ingredient_array):
    count = 0
    for ingredient in ingredient_array:
        if ingredient in ingredient_mapping['southern_us']:
            count += 1
    return (count / float(len(ingredient_array)))

def getFilipinoScore(ingredient_array):
    count = 0
    for ingredient in ingredient_array:
        if ingredient in ingredient_mapping['filipino']:
            count += 1
    return (count / float(len(ingredient_array)))

def getIndianScore(ingredient_array):
    count = 0
    for ingredient in ingredient_array:
        if ingredient in ingredient_mapping['indian']:
            count += 1
    return (count / float(len(ingredient_array)))

def getJamaicanScore(ingredient_array):
    count = 0
    for ingredient in ingredient_array:
        if ingredient in ingredient_mapping['jamaican']:
            count += 1
    return (count / float(len(ingredient_array)))

def getSpanishScore(ingredient_array):
    count = 0
    for ingredient in ingredient_array:
        if ingredient in ingredient_mapping['spanish']:
            count += 1
    return (count / float(len(ingredient_array)))


def getItalianScore(ingredient_array):
    count = 0
    for ingredient in ingredient_array:
        if ingredient in ingredient_mapping['italian']:
            count += 1
    return (count / float(len(ingredient_array)))

def getGreekScore(ingredient_array):
    count = 0
    for ingredient in ingredient_array:
        if ingredient in ingredient_mapping['greek']:
            count += 1
    return (count / float(len(ingredient_array)))

def getGreekScore(ingredient_array):
    count = 0
    for ingredient in ingredient_array:
        if ingredient in ingredient_mapping['greek']:
            count += 1
    return (count / float(len(ingredient_array)))

def getMexicanScore(ingredient_array):
    count = 0
    for ingredient in ingredient_array:
        if ingredient in ingredient_mapping['mexican']:
            count += 1
    return (count / float(len(ingredient_array)))

def getChineseScore(ingredient_array):
    count = 0
    for ingredient in ingredient_array:
        if ingredient in ingredient_mapping['chinese']:
            count += 1
    return (count / float(len(ingredient_array)))

def getBritishScore(ingredient_array):
    count = 0
    for ingredient in ingredient_array:
        if ingredient in ingredient_mapping['british']:
            count += 1
    return (count / float(len(ingredient_array)))

def getThaiScore(ingredient_array):
    count = 0
    for ingredient in ingredient_array:
        if ingredient in ingredient_mapping['thai']:
            count += 1
    return (count / float(len(ingredient_array)))

def getViatnameseScore(ingredient_array):
    count = 0
    for ingredient in ingredient_array:
        if ingredient in ingredient_mapping['vietnamese']:
            count += 1
    return (count / float(len(ingredient_array)))

def getCCScore(ingredient_array):
    count = 0
    for ingredient in ingredient_array:
        if ingredient in ingredient_mapping['cajun_creole']:
            count += 1
    return (count / float(len(ingredient_array)))

def getbrazilianScore(ingredient_array):
    count = 0
    for ingredient in ingredient_array:
        if ingredient in ingredient_mapping['brazilian']:
            count += 1
    return (count / float(len(ingredient_array)))

def getfrenchScore(ingredient_array):
    count = 0
    for ingredient in ingredient_array:
        if ingredient in ingredient_mapping['french']:
            count += 1
    return (count / float(len(ingredient_array)))

def getjapaneseScore(ingredient_array):
    count = 0
    for ingredient in ingredient_array:
        if ingredient in ingredient_mapping['japanese']:
            count += 1
    return (count / float(len(ingredient_array)))

def getIrishScore(ingredient_array):
    count = 0
    for ingredient in ingredient_array:
        if ingredient in ingredient_mapping['irish']:
            count += 1
    return (count / float(len(ingredient_array)))


def getKoreanScore(ingredient_array):
    count = 0
    for ingredient in ingredient_array:
        if ingredient in ingredient_mapping['korean']:
            count += 1
    return (count / float(len(ingredient_array)))


def getMoroccanScore(ingredient_array):
    count = 0
    for ingredient in ingredient_array:
        if ingredient in ingredient_mapping['moroccan']:
            count += 1
    return (count / float(len(ingredient_array)))

def getRussianScore(ingredient_array):
    count = 0
    for ingredient in ingredient_array:
        if ingredient in ingredient_mapping['russian']:
            count += 1
    return (count / float(len(ingredient_array)))

df_train['length'] = df_train['ingredients'].apply(len)
df_test['length'] = df_test['ingredients'].apply(len)
plt.figure(figsize=(15,7))
sns.distplot(df_train['length'], bins = 50, color = 'red')
plt.figure(figsize=(15,7))
sns.distplot(np.log(df_train['length']), bins = 20, color = 'blue')
# %%time

# df_train[['greek', 'southern_us', 'filipino', 'indian', 'jamaican', 'spanish', 'italian', 'mexican', 'chinese', 'british', 'thai', 'vietnamese', 'cajun_creole', 'brazilian', 'french', 'japanese', 'irish', 'korean', 'moroccan', 'russian']] = df_train['ingredients'].apply([getGreekScore,getsouthern_usScore,getFilipinoScore, getIndianScore, getJamaicanScore, getSpanishScore, getItalianScore, getMexicanScore, getChineseScore, getBritishScore, getThaiScore, getViatnameseScore, getCCScore, getbrazilianScore, getfrenchScore, getjapaneseScore, getIrishScore, getKoreanScore, getMoroccanScore, getRussianScore])
# df_test[['greek', 'southern_us', 'filipino', 'indian', 'jamaican', 'spanish', 'italian', 'mexican', 'chinese', 'british', 'thai', 'vietnamese', 'cajun_creole', 'brazilian', 'french', 'japanese', 'irish', 'korean', 'moroccan', 'russian']] = df_test['ingredients'].apply([getGreekScore,getsouthern_usScore,getFilipinoScore, getIndianScore, getJamaicanScore, getSpanishScore, getItalianScore, getMexicanScore, getChineseScore, getBritishScore, getThaiScore, getViatnameseScore, getCCScore, getbrazilianScore, getfrenchScore, getjapaneseScore, getIrishScore, getKoreanScore, getMoroccanScore, getRussianScore])
df_train.head()
def ingToString(ingredient_list):
    return ' '.join(ingredient_list)
%%time

df_train['strIngredient'] = df_train['ingredients'].apply(ingToString)
df_test['strIngredient'] = df_test['ingredients'].apply(ingToString)

key = df_test['id']
df_train = df_train.drop(columns=['id', 'ingredients'], axis = 1)
df_test = df_test.drop(columns=['id', 'ingredients'], axis = 1)
df_train.head()
from nltk.corpus import stopwords
import string

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

pipeline2 = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', ExtraTreesClassifier()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

pipeline3 = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', SVC()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
X_train, X_test, y_train, y_test = train_test_split(df_train['strIngredient'],df_train['cuisine'], test_size = 0.2, shuffle = True)
X_train.head()
pipeline3.fit(X_train, y_train)
predictions = pipeline2.predict(X_test)
print (classification_report(y_test, predictions))
pipeline3.fit(df_train['strIngredient'],df_train['cuisine'])
final_predictions = pipeline3.predict(df_test['strIngredient'])

df = pd.DataFrame({"id":key, "cuisine": final_predictions})
df.to_csv("mySubmission.csv", index = False)
len(key)

