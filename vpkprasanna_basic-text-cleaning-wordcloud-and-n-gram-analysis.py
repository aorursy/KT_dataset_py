# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# pd.set_option('display.max_colwidth', -1)

import matplotlib.pyplot as plt

from matplotlib import rcParams

import re

from wordcloud import WordCloud

from collections import Counter

import csv

from matplotlib import rcParams

from nltk.corpus import stopwords

import nltk

from nltk.util import ngrams

stop = stopwords.words('english')

from sklearn.feature_extraction.text import CountVectorizer

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression



from sklearn.metrics import accuracy_score, confusion_matrix,classification_report,plot_confusion_matrix

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
true = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")

false = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")

true.head()
false.head()
true.subject.value_counts()
rcParams['figure.figsize'] = 15,10

true.subject.value_counts().plot(kind="bar")
rcParams['figure.figsize'] = 15,10

false.subject.value_counts().plot(kind="bar")
politics = true[true['subject']=="politicsNews"]

worldnews = true[true['subject']=="worldnews"]

print(politics.shape)

print(worldnews.shape)
politics_text_len = politics['text'].str.len()

worldnews_text_len = worldnews['text'].str.len()
print("The maximum lenght of string in Politcs news is {} words".format(max(politics_text_len)))

print("The maximum lenght of string in World news is {} words".format(max(worldnews_text_len)))
def tokenizeandstopwords(text):

    tokens = nltk.word_tokenize(text)

    # taken only words (not punctuation)

    token_words = [w for w in tokens if w.isalpha()]

    meaningful_words = [w for w in token_words if not w in stop]

    joined_words = ( " ".join(meaningful_words))

    return joined_words
politics['text'] = politics['text'].apply(tokenizeandstopwords)

worldnews['text'] = worldnews['text'].apply(tokenizeandstopwords)
def generate_word_cloud(text):

    wordcloud = WordCloud(

        width = 3000,

        height = 2000,

        background_color = 'black').generate(str(text))

    fig = plt.figure(

        figsize = (40, 30),

        facecolor = 'k',

        edgecolor = 'k')

    plt.imshow(wordcloud, interpolation = 'bilinear')

    plt.axis('off')

    plt.tight_layout(pad=0)

    plt.show()

politics_text = politics.text.values

generate_word_cloud(politics_text)

worldnews_text = worldnews.text.values

generate_word_cloud(worldnews_text)
false.head()
set(false.subject)
Government_News = false[false['subject']=="Government News"]

Middle_east = false[false['subject']=="Middle-east"]

News = false[false['subject']=="News"]

US_News = false[false['subject']=="US_News"]

politics = false[false['subject']=="politics"]
Government_News['text'] = Government_News['text'].apply(tokenizeandstopwords)

Middle_east['text'] = Middle_east['text'].apply(tokenizeandstopwords)

News['text'] = News['text'].apply(tokenizeandstopwords)

US_News['text'] = US_News['text'].apply(tokenizeandstopwords)

politics['text'] = politics['text'].apply(tokenizeandstopwords)
govertment_news_text = Government_News['text'].values

generate_word_cloud(govertment_news_text)
middleast_news_text = Middle_east['text'].values

generate_word_cloud(middleast_news_text)
news_text = News['text'].values

generate_word_cloud(news_text)
usnews_text = US_News['text'].values

generate_word_cloud(usnews_text)
politicsFake_text = politics['text'].values

generate_word_cloud(politicsFake_text)
false['target'] = 'fake'

true['target'] = 'true'

news = pd.concat([false, true]).reset_index(drop = True)

news.head()
news.shape
news['text'] = news['text'].apply((lambda y:re.sub("http://\S+"," ", y)))

news['text'] = news['text'].apply((lambda x:re.sub("\@", " ",x.lower())))
news.head()
def basic_clean(text):

  """

  A simple function to clean up the data. All the words that

  are not designated as a stop word is then lemmatized after

  encoding and basic regex parsing are performed.

  """

  wnl = nltk.stem.WordNetLemmatizer()

  stopwords = nltk.corpus.stopwords.words('english')

  words = re.sub(r'[^\w\s]', '', text).split()

  return [wnl.lemmatize(word) for word in words if word not in stopwords]



true_word = basic_clean(''.join(str(true['text'].tolist())))
true_bigrams_series = (pd.Series(nltk.ngrams(true_word, 2)).value_counts())[:20]

true_bigrams_series.sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))

plt.title('20 Most Frequently Occuring Bigrams')

plt.ylabel('Bigram')

plt.xlabel('# of Occurances')
true_trigrams_series = (pd.Series(nltk.ngrams(true_word, 3)).value_counts())[:20]

true_trigrams_series.sort_values().plot.barh(color='red', width=.9, figsize=(12, 8))

plt.title('20 Most Frequently Occuring Trigrams')

plt.ylabel('Trigram')

plt.xlabel('# of Occurances')
false_word = basic_clean(''.join(str(false['text'].tolist())))
flase_bigrams_series = (pd.Series(nltk.ngrams(false_word, 2)).value_counts())[:20]
flase_bigrams_series.sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))

plt.title('20 Most Frequently Occuring Bigrams')

plt.ylabel('Bigram')

plt.xlabel('# of Occurances')
false_trigrams_series = (pd.Series(nltk.ngrams(false_word, 3)).value_counts())[:20]

false_trigrams_series.sort_values().plot.barh(color='red', width=.9, figsize=(12, 8))

plt.title('20 Most Frequently Occuring Trigrams')

plt.ylabel('Trigram')

plt.xlabel('# of Occurances')
words = basic_clean(''.join(str(news['text'].tolist())))
bigrams_series = (pd.Series(nltk.ngrams(words, 2)).value_counts())[:20]
bigrams_series.sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))

plt.title('20 Most Frequently Occuring Bigrams')

plt.ylabel('Bigram')

plt.xlabel('# of Occurances')
trigrams_series = (pd.Series(nltk.ngrams(words, 3)).value_counts())[:20]
trigrams_series.sort_values().plot.barh(color='red', width=.9, figsize=(12, 8))

plt.title('20 Most Frequently Occuring Trigrams')

plt.ylabel('Trigram')

plt.xlabel('# of Occurances')


x_train,x_test,y_train,y_test = train_test_split(news['text'], news.target, test_size=0.2, random_state=2020)



pipe = Pipeline([('vect', CountVectorizer()),

                 ('tfidf', TfidfTransformer()),

                 ('model', LogisticRegression())])



model = pipe.fit(x_train, y_train)

prediction = model.predict(x_test)

print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
print(confusion_matrix(y_test, prediction))

print(classification_report(y_test, prediction))
plot_confusion_matrix(model,x_test,y_test)