import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing the libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# Importing the dataset

dataset = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')

dataset.head(15)
# Distribution of data

dataset['sentiment'].value_counts()
# Tokenize the text

from nltk.tokenize.toktok import ToktokTokenizer

tokenizer = ToktokTokenizer()
# Import BeautifulSoup which helps us act with html text

from bs4 import BeautifulSoup

def html_remove(review):

    return BeautifulSoup(review, 'html.parser').get_text()



dataset['review'] = dataset['review'].apply(html_remove)
# Import required library

import re

def non_alpha_numeric_remove(review):

    return re.sub(pattern = '[^a-zA-Z0-9]', repl = ' ', string = review)

dataset['review'] = dataset['review'].apply(non_alpha_numeric_remove)
# Import the required library

from nltk.stem import PorterStemmer

ps = PorterStemmer()

def stemming(review):

    rev = ' '.join([ps.stem(word) for word in review.split()])

    return rev

dataset['review'] = dataset['review'].apply(stemming)
def to_lower(review):

    return review.lower()

dataset['review'] = dataset['review'].apply(to_lower)
# Importing stopwords from nltk

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

print(stopwords.words('english'))

def remove_stopwords(review):

    tokens = tokenizer.tokenize(review)

    tokens = [token.strip() for token in tokens]

    new_tokens = [token for token in tokens if token not in stopwords.words('english')]

    new_review = ' '.join(new_tokens)

    return new_review

dataset['review'] = dataset['review'].apply(remove_stopwords)
dataset.to_csv('Cleaned dataset.csv')
X_train = dataset.iloc[:40000, 0].values

X_test = dataset.iloc[40000:, 0].values

y_train = dataset.iloc[:40000, 1].values

y_test = dataset.iloc[40000:, 1].values
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary = False, ngram_range = (1,3))

X_train = cv.fit_transform(X_train)

X_test = cv.transform(X_test)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y_train = le.fit_transform(y_train)

y_test = le.transform(y_test)
# Training our model using Logistic Regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter = 500, random_state = 0)

lr.fit(X_train, y_train)
# Prediction using our model

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

lr_predictions = lr.predict(X_test)

print('Accuracy of Logistic Regression is: ', accuracy_score(y_test, lr_predictions) * 100)

print(classification_report(y_test,lr_predictions))
import seaborn as sns

plt.figure(figsize=(9,9))

sns.heatmap(confusion_matrix(y_test, lr_predictions), annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'YlGnBu');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Accuracy Score: {0}'.format(accuracy_score(y_test, lr_predictions) * 100)

plt.title(all_sample_title, size = 15);
# Training our model using Multinomial Naive Bayes

from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()

mnb.fit(X_train, y_train)
# Prediction using our model

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

mnb_predictions = mnb.predict(X_test)

print('Accuracy of Multinomial Naive Bayes is: ', accuracy_score(y_test, mnb_predictions) * 100)

print(classification_report(y_test, mnb_predictions))
import seaborn as sns

plt.figure(figsize=(9,9))

sns.heatmap(confusion_matrix(y_test, mnb_predictions), annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'YlGnBu');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Accuracy Score: {0}'.format(accuracy_score(y_test, mnb_predictions) * 100)

plt.title(all_sample_title, size = 15);