# This Python 3 environment comes with many helpful analytics libraries installed



import pandas as pd

import numpy as np

import pyodbc

from pandas import DataFrame



import string

import nltk

from sklearn import re

from nltk.corpus import stopwords

from nltk.classify import SklearnClassifier

from nltk.stem import PorterStemmer

from nltk import FreqDist 

from nltk.stem.porter import *

import re

from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from wordcloud import WordCloud

import matplotlib.pyplot as plt

import seaborn as sns

import time

import csv

import datetime

from datetime import datetime

from datetime import date, timedelta

from plotly import tools

import plotly.offline as py

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



#Model Selection and Validation

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import confusion_matrix, classification_report, f1_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
train = train.astype({"id" : int, "target" : int, "text" : str})

train.head()
df_combined = train.append(test, ignore_index=True, sort = False)

df_combined = df_combined.astype({"text" : str})

df_combined.head()
df_combined = df_combined.loc[df_combined["keyword"].notnull()]

df_combined = df_combined.loc[df_combined["text"].notnull()]

df_combined.shape
sns.barplot(y=train['keyword'].value_counts()[:30].index,x=train['keyword'].value_counts()[:30],orient='h')

plt.figure(figsize=(26, 18))
# REMOVE @User



def remove_pattern(input_text, pattern):

    r = re.findall(pattern, input_text)

    for i in r:

        input_text = re.sub(i, '', input_text)

        

    return input_text  
df_combined['tidy_tweet'] = np.vectorize(remove_pattern)(df_combined['text'], "@[\w]*")

df_combined['tidy_tweet'] = df_combined['tidy_tweet'].str.replace('[^\w\d#\s]',' ')

df_combined['tidy_tweet'] = df_combined['tidy_tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))

df_combined.tail()
#REMOVE Stop words



stop = stopwords.words('english')

df_combined["tidy_tweet"] = df_combined["tidy_tweet"].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

df_combined['tidy_tweet'] = df_combined['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

df_combined.head()
df_train_disaster = df_combined.loc[df_combined["target"] == 1.0]

df_train_nondisaster = df_combined.loc[df_combined["target"] == 0.0]

df_test = df_combined.loc[df_combined["target"].isnull()]

df_train_disaster.tail()
seperated_words_dis = df_train_disaster['tidy_tweet'].apply(lambda x: x.split())

seperated_words_dis
# from nltk.corpus import words

# sample = []

# output = []

# word = words.words()

# x = np.array(word)

# word = np.unique(x)



# for text in seperated_words_dis:

#     for j in text:

#         for k in word:

#             if ([j]) == ([k]):

#                 sample.append(j)

#     output.append(sample)

#     sample = []

# output



# from nltk.tokenize.treebank import TreebankWordDetokenizer

# output1 = []

# for i in output:

#     output1 = [' '.join(x) for x in output]

# output1



# disaster_tweet = pd.DataFrame(output1)

# disaster_tweet = disaster_tweet.rename(columns = {0 : "Tweet_tweeked"})

# df_train_disaster = df_train_disaster.reset_index()

# df_train_disaster = df_train_disaster.astype({"id" : int, "target" : int})

# df_train_disaster = df_train_disaster.merge(disaster_tweet, how = 'inner', left_index = True, right_index = True)

# df_train_disaster = df_train_disaster.filter(["id", "keyword", "location", "text", "target", "Tweet_tweeked"])

# df_train_disaster.head()
train_disaster = pd.read_csv("../input/twitter-competition/train_disaster.csv")

train_disaster = train_disaster.filter(["id", "keyword", "location", "text", "target", "Tweet_tweeked"])



train_nondisaster = pd.read_csv("../input/twitter-competition/train_nondisaster.csv")

train_nondisaster = train_nondisaster.filter(["id", "keyword", "location", "text", "target", "Tweet_tweeked"])



tweeked_test = pd.read_csv("../input/twitter-competition/tweeked_test.csv")

tweeked_test = tweeked_test.filter(["id", "keyword", "location", "text", "Tweet_tweeked"])
train_disaster = train_disaster.filter(["id", "keyword", "location", "text", "target", "Tweet_tweeked"])

train_disaster = train_disaster.astype({"id": int, "target": int})

train_disaster = train_disaster.loc[train_disaster["Tweet_tweeked"] != "none"]

train_disaster = train_disaster.loc[train_disaster["Tweet_tweeked"].notnull()]

train_disaster["target"] = train_disaster["target"].replace({1: "real", 0 : "not_real"})

train_disaster.shape
train_nondisaster = train_nondisaster.filter(["id", "keyword", "location", "text", "target", "Tweet_tweeked"])

train_nondisaster = train_nondisaster.astype({"id": int, "target": int})

train_nondisaster = train_nondisaster.loc[train_nondisaster["Tweet_tweeked"] != "none"]

train_nondisaster = train_nondisaster.loc[train_nondisaster["Tweet_tweeked"].notnull()]

train_nondisaster["target"] = train_nondisaster["target"].replace({1: "real", 0 : "not_real"})

train_nondisaster.shape
tweeked_test = tweeked_test.filter(["id", "keyword", "location", "text", "Tweet_tweeked"])

tweeked_test = tweeked_test.astype({"id": int})

# tweeked_test = tweeked_test.loc[tweeked_test["Tweet_tweeked"] != "none"]

# tweeked_test = tweeked_test.loc[tweeked_test["Tweet_tweeked"].notnull()]

tweeked_test.shape
#Train Disaster Data:



comment_words = ' '

# iterate through the csv file 

for val in train_disaster.Tweet_tweeked: 

    # typecaste each val to string 

    val = str(val)

    # split the value 

    tokens = val.split()  

    for words in tokens: 

        comment_words = comment_words + words + ' '



wordcloud = WordCloud(width = 700, height = 700, background_color ='black', min_font_size = 8).generate(comment_words) 

  

# plot the WordCloud image                        

plt.figure(figsize = (7, 7), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show()
ax = train_disaster['keyword'].value_counts().sort_values().plot(kind='barh',

                                    figsize=(15,40),

                                    title="Number for each Keyword")

ax.set_xlabel("Keywords")

ax.set_ylabel("Frequency")
#Train Non-Disaster Data:



comment_words = ' ' 

# iterate through the csv file 

for val in train_nondisaster.Tweet_tweeked: 

    # typecaste each val to string 

    val = str(val)

    # split the value 

    tokens = val.split()  

    for words in tokens: 

        comment_words = comment_words + words + ' '



wordcloud = WordCloud(width = 700, height = 700, background_color ='black', min_font_size = 8).generate(comment_words) 

  

# plot the WordCloud image                        

plt.figure(figsize = (7, 7), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show()
ax = train_nondisaster['keyword'].value_counts().sort_values().plot(kind='barh',

                                    figsize=(15,40),

                                    title="Number for each Keyword")

ax.set_xlabel("Keywords")

ax.set_ylabel("Frequency")
#Test Data:



comment_words = ' '

# iterate through the csv file 

for val in tweeked_test.Tweet_tweeked: 

    # typecaste each val to string 

    val = str(val)

    # split the value 

    tokens = val.split()  

    for words in tokens: 

        comment_words = comment_words + words + ' '



wordcloud = WordCloud(width = 700, height = 700, background_color ='black', min_font_size = 8).generate(comment_words) 

  

# plot the WordCloud image                        

plt.figure(figsize = (7, 7), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show()
ax = tweeked_test['keyword'].value_counts().sort_values().plot(kind='barh',

                                    figsize=(15,40),

                                    title="Number for each Keyword")

ax.set_xlabel("Keywords")

ax.set_ylabel("Frequency")
#Combine the Train Disaster and Non-Disaster Data:

train = train_disaster.append(train_nondisaster, ignore_index = True)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(train.Tweet_tweeked).toarray()

labels = train.target

features.shape
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB



X_train, X_test, y_train, y_test = train_test_split(train['Tweet_tweeked'], train['target'], random_state = 0)



count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)



tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)



clf = MultinomialNB().fit(X_train_tfidf, y_train)
clf.predict(count_vect.transform(X_test))
models = [

    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),

    LinearSVC(),

    MultinomialNB(),

    LogisticRegression(random_state=0)]



CV = 5

cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []

for model in models:

    model_name = model.__class__.__name__

    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)

    for fold_idx, accuracy in enumerate(accuracies):

        entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

cv_df
import seaborn as sns

sns.boxplot(x='model_name', y='accuracy', data=cv_df)

sns.stripplot(x='model_name', y='accuracy', data=cv_df, 

              size=8, jitter=True, edgecolor="gray", linewidth=2)

plt.show()
cv_df.groupby('model_name').accuracy.mean()
model = LinearSVC()

X1_train, X1_test, y1_train, y1_test, indices_train, indices_test = train_test_split(features, labels, train.index, test_size=0.33, random_state=0)

model.fit(X1_train, y1_train)

y1_pred = model.predict(X1_test)



from sklearn.metrics import confusion_matrix



conf_mat = confusion_matrix(y1_test, y1_pred)



from sklearn import metrics

print(metrics.classification_report(y1_test, y1_pred, target_names=train['target'].unique()))
sample_submission
tweeked_test = tweeked_test.filter(["id", "keyword", "location", "text", "Tweet_tweeked"])

y_pre=clf.predict(count_vect.transform(tweeked_test["Tweet_tweeked"].values.astype('U')))

y_pre = pd.DataFrame(y_pre)

sub = tweeked_test.merge(y_pre, how = "inner", left_index = True, right_index = True)

sub = sub.rename(columns = { 0 : 'Target'})

sub["Target"] = sub["Target"].replace({"real" : 1, "not_real": 0})

Final_sub = sub.filter(["id", "Target"])

print(Final_sub)

Final_sub.to_csv('submission.csv',index=False)