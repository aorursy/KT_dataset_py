import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

import plotly 

import plotly.graph_objects as go

import cufflinks as cf

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

cf.go_offline()
df = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')

df.head()
# see the null data here

df.isnull().sum()
df.info()
df.drop(['Unnamed: 2','Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
# search the most relevant message 

df['v2'].describe()
# count of ham and spam

df['v1'].value_counts()
# convert categorical v1 to numerical with new column

df['v1_nm'] = df.v1.map({'ham':0, 'spam':1})

df.head()
# interactive plotly hist plot for numerical vi_nm columns(i,e ham and spam)

df['v1_nm'].iplot(kind='hist')
# creating a new column with message length using v2 column

df['v2_le'] = df.v2.apply(len)

df.head()
# Histogram plot for spam and ham labels with respeect to message length

plt.figure(figsize=(12,8))

df[df['v1']=='ham'].v2_le.plot(bins = 50, kind= 'hist', color='blue', label='ham', alpha=0.75)

df[df['v1']=='spam'].v2_le.plot(bins=50, kind= 'hist', color='red', label = 'spam', alpha=0.75)

plt.legend()

plt.xlabel('Message length')
# describe the ham for some numerical insights

df[df['v1']=='ham'].describe()
# describe the spam some numerical insights

df[df['v1']=='spam'].describe()
# describe the both numerical columns

df.describe()
# see in describe we have 910 word message, let's look at it

df[df['v2_le']==910].v2.iloc[0]
import string 

from nltk.corpus import stopwords



def text_process(mess):

    """

    Takes in a string of text, then performs the following:

    1. Remove all punctuation

    2. Remove all stopwords

    3. Returns a list of the cleaned text

    """

    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']

    

    # Check characters to see if they are in punctuation

    nopunc = [char for char in mess if char not in string.punctuation]

    

    # Join the characters again to form the string.

    nopunc = ''.join(nopunc)

    

    # Now just remove any stopwords

    return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])
df['clean_msg'] = df.v2.apply(text_process)
df.head()
type(stopwords.words('english'))
from collections import Counter



words = df[df['v1']=='ham'].clean_msg.apply(lambda x: [word.lower() for word in x.split()])

ham_words = Counter()



for msg in words:

    ham_words.update(msg)

    

print(ham_words.most_common(50))    
words = df[df.v1=='spam'].clean_msg.apply(lambda x: [word.lower() for word in x.split()])

spam_words = Counter()



for msg in words:

    spam_words.update(msg)

    

print(spam_words.most_common(50))
# how to define X and y (from the SMS data) for use with COUNTVECTORIZER

X = df.clean_msg

y = df.v1_nm

print(X.shape)

print(y.shape)
# split X and y into training and testing sets 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.feature_extraction.text import CountVectorizer



# instantiate the vectorizer

vect = CountVectorizer()

vect.fit(X_train)
# learn training data vocabulary, then use it to create a document-term matrix

X_train_dtm = vect.transform(X_train)







# equivalently: combine fit and transform into a single step

X_train_dtm = vect.fit_transform(X_train)



# examine the document-term matrix

X_train_dtm
# transform testing data (using fitted vocabulary) into a document-term matrix

X_test_dtm = vect.transform(X_test)

X_test_dtm
from sklearn.feature_extraction.text import TfidfTransformer



tfidf_transformer = TfidfTransformer()

tfidf_transformer.fit(X_train_dtm)

tfidf_transformer.transform(X_train_dtm)
# import and instantiate a Multinomial Naive Bayes model

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
# train the model using X_train_dtm (timing it with an IPython "magic command")

%time nb.fit(X_train_dtm, y_train)
# make class predictions for X_test_dtm

y_pred_class = nb.predict(X_test_dtm)
# calculate accuracy of class predictions

from sklearn import metrics

metrics.accuracy_score(y_test, y_pred_class)
# print the confusion matrix

metrics.confusion_matrix(y_test, y_pred_class)
# print message text for false positives (ham incorrectly classifier)

# X_test[(y_pred_class==1) & (y_test==0)]

X_test[y_pred_class > y_test]
# print message text for false negatives (spam incorrectly classifier)

X_test[y_pred_class < y_test]
# example of false negative 

X_test[4949]
# calculate predicted probabilities for X_test_dtm (poorly calibrated)

y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]

y_pred_prob
# calculate AUC

metrics.roc_auc_score(y_test, y_pred_prob)
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline



pipe = Pipeline([('bow', CountVectorizer()), 

                 ('tfid', TfidfTransformer()),  

                 ('model', MultinomialNB())])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
metrics.accuracy_score(y_test, y_pred)
metrics.confusion_matrix(y_test, y_pred)
# import an instantiate a logistic regression model

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='liblinear')
# train the model using X_train_dtm

%time logreg.fit(X_train_dtm, y_train)
# make class predictions for X_test_dtm

y_pred_class = logreg.predict(X_test_dtm)
# calculate predicted probabilities for X_test_dtm (well calibrated)

y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]

y_pred_prob
# calculate accuracy

metrics.accuracy_score(y_test, y_pred_class)
metrics.confusion_matrix(y_test, y_pred_class)
# calculate AUC

metrics.roc_auc_score(y_test, y_pred_prob)
# show default parameters for CountVectorizer

vect
# remove English stop words

vect = CountVectorizer(stop_words='english')
# include 1-grams and 2-grams

vect = CountVectorizer(ngram_range=(1, 2))
# ignore terms that appear in more than 50% of the documents

vect = CountVectorizer(max_df=0.5)
# only keep terms that appear in at least 2 documents

vect = CountVectorizer(min_df=2)