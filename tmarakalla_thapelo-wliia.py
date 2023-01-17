# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegressionCV,SGDClassifier
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.svm import SVC
import os
print(os.listdir("../input"))
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
train_df.head()
train_df['president'].value_counts().plot(kind = 'bar')
plt.show()
train_data = []
for i, row in train_df.iterrows():
    for text in row['text'].split('.'):
        train_data.append([row['president'], text])
train_data = pd.DataFrame(train_data, columns=['president', 'text'])
train_data['president'].value_counts().plot(kind = 'bar')
plt.show()
train_data.head()
def remove_punctuation_numbers(text):
    punc_numbers = string.punctuation + '0123456789'
    return ''.join([l for l in text if l not in punc_numbers])
def tokeniser(text):
    return TreebankWordTokenizer().tokenize(text)
def lemmetizer(tokens):
    wordnet_lemmatizer = WordNetLemmatizer()
    return [wordnet_lemmatizer.lemmatize(word) for word in tokens]
def remove_stop_words(tokens):
    return [t for t in tokens if t not in set(stopwords.words('english'))]
def data_cleaner(text):
    text = text.lower()
    text = remove_punctuation_numbers(text)
    lst = tokeniser(text)
    lst = remove_stop_words(lst)
    return ' '.join(lemmetizer(lst))
train_data['clean_text'] = train_data['text'].apply(data_cleaner)
train_data.head()
for pres in train_data['president'].unique():
    words =[]
    for sentence in train_data[train_data['president'] == pres].clean_text:
        words.extend(tokeniser(sentence))
    
    wordcloud = WordCloud().generate_from_frequencies(frequencies=Counter(words))
    plt.figure(figsize=(12,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(pres)
    plt.show()
train_data.president = train_data.president.map({'deKlerk':0,'Mandela':1,
                                                'Mbeki':2, 'Motlanthe':3,
                                                'Zuma': 4, 'Ramaphosa':5})
X = train_data.clean_text
y = train_data.president
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=0,
                                                    stratify=y)
vect = CountVectorizer(ngram_range=(1,2))
X_train_ = vect.fit_transform(X_train)
log = LogisticRegressionCV(dual=False, penalty='l2', multi_class='multinomial')
log.fit(X_train_, y_train)
print(accuracy_score(y_train, log.predict(X_train_)))
print(accuracy_score(y_test, log.predict(vect.transform(X_test))))
test_data = pd.read_csv('../input/test.csv')
test_data.head()
test_data.text = test_data.text.apply(data_cleaner)
test_data['president'] = log.predict(vect.transform(test_data.text))
test_data.drop('text', axis=1, inplace=True,)
test_data.head()
test_data.to_csv('Thapelo_log.csv',index=False)
