import pandas as pd

import numpy as np

import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from collections import  Counter

import string

from nltk.corpus import stopwords

import nltk

from sklearn.naive_bayes import MultinomialNB

from xgboost import XGBClassifier

from sklearn.metrics import f1_score



from sklearn import model_selection

from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier

from sklearn.svm import SVC
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train.head()
test.head()
train.shape, test.shape
# tokenizer = nltk.tokenize.WhitespaceTokenizer()

# tokenizer = nltk.tokenize.TreebankWordTokenizer()

# tokenizer = nltk.tokenize.WordPunctTokenizer()

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')



normalization = None

# normalization = 'stemmer'

# normalization = 'lemmatizer'



vectorizer = 'countvectorizer'

# vectorizer = 'tfidfvectorizer'
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'', text)



train['text'] = train['text'].apply(remove_URL)

test['text'] = test['text'].apply(remove_URL)
train.head()
def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



train['text'] = train['text'].apply(remove_html)

test['text'] = test['text'].apply(remove_html)
def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



train['text'] = train['text'].apply(remove_emoji)

test['text'] = test['text'].apply(remove_emoji)
def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



train['text'] = train['text'].apply(remove_punct)

test['text'] = test['text'].apply(remove_punct)
# Before Tokenization

train.head(3)
train['text'] = train['text'].apply(tokenizer.tokenize)

test['text'] = test['text'].apply(tokenizer.tokenize)
# After Tokenization

train.head()
def stem_tokens(tokens):

    stemmer = nltk.stem.PorterStemmer()

    tokens = [stemmer.stem(token) for token in tokens]

    return tokens



def lemmatize_tokens(tokens):

    lemmatizer = nltk.stem.WordNetLemmatizer()

    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens



def normalize_tokens(normalization):

    if normalization is not None:

        if normalization == 'stemmer':

            train['text'] = train['text'].apply(stem_tokens)

        elif normalization == 'lemmatizer':

            train['text'] = train['text'].apply(lemmatize_tokens)

        

normalize_tokens(normalization)
train.head()
def remove_stopwords(text):

    words = [w for w in text if w not in stopwords.words('english')]

    return words



train['text'] = train['text'].apply(remove_stopwords)

test['text'] = test['text'].apply(remove_stopwords)
train.head()
def combine_tokens(text):

    combined_text = ' '.join(text)

    return combined_text



train['text'] = train['text'].apply(combine_tokens)

test['text'] = test['text'].apply(combine_tokens)
train.head()
# Vectorization

def vectorize(vectorizer):

    if vectorizer == 'countvectorizer':

        print('countvectorizer')

        vectorizer = CountVectorizer()

        train_vectors = vectorizer.fit_transform(train['text'])

        test_vectors = vectorizer.transform(test['text'])

    elif vectorizer == 'tfidfvectorizer':

        print('tfidfvectorizer')

        vectorizer = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))

        train_vectors = vectorizer.fit_transform(train['text'])

        test_vectors = vectorizer.transform(test['text'])

    return train_vectors, test_vectors



train_vectors, test_vectors = vectorize(vectorizer)
train_vectors[0].todense()
# Fitting a simple Naive Bayes on Counts

clf_NB = MultinomialNB()

scores = model_selection.cross_val_score(clf_NB, train_vectors, train["target"], cv=5, scoring="f1")

scores
# clf_NB.fit(train_vectors, train['target'])
svc = SVC(kernel='rbf', C=70, gamma='auto', probability=True, random_state=41)

rfc = RandomForestClassifier(n_estimators=200, random_state=41)

gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.2, random_state=41)
vcf = VotingClassifier(estimators=[('svc', svc), ('rfc', rfc), ('gbc', gbc)], voting='soft')

vcf.fit(train_vectors, train['target'])
def make_submission(submission_file, model, test_vectors):

    sample_submission = pd.read_csv(submission_file)

    sample_submission['target'] = model.predict(test_vectors)

    sample_submission.to_csv('my_submission.csv', index=False)
submission_file = '/kaggle/input/nlp-getting-started/sample_submission.csv'

test_vectors = test_vectors

make_submission(submission_file, vcf, test_vectors)