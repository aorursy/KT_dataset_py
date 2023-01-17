import sys, os, re 

import numpy as np  

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from sklearn.metrics import mean_absolute_error, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix



from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score





from nltk import sent_tokenize, word_tokenize

from nltk.corpus import stopwords

from nltk import WordNetLemmatizer





from scipy.sparse import hstack

import xgboost as xgb





from nltk.tokenize.treebank import TreebankWordDetokenizer



def clean_text(text):

    """

    This function takes as input a text on which several 

    NLTK algorithms will be applied in order to preprocess it

    """

    tokens = word_tokenize(text)

    # Remove the punctuations

    tokens = [word for word in tokens if word.isalpha()]

    # Remove stopword

    tokens = [word for word in tokens if not word in stopwords.words("english")]

    # Lower the tokens

    tokens = [word.lower() for word in tokens]

    

    # Lemmatize

    lemma = WordNetLemmatizer()

    tokens = [lemma.lemmatize(word, pos = "v") for word in tokens]

    tokens = [lemma.lemmatize(word, pos = "n") for word in tokens]

    

    return TreebankWordDetokenizer().detokenize(tokens)
# Read the data

df_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv', index_col='id')

df_test  = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv', index_col='id')



df_train['text_cleaned'] = [clean_text(texto) for texto in df_train['text']]

df_test['text_cleaned'] = [clean_text(texto) for texto in df_test['text']]



class_names = ['target']



train = df_train.fillna(' ')

test = df_test.fillna(' ')

train.head()
#train_text = train['text']

#test_text = test['text']

train_text = train['text_cleaned']

test_text = test['text_cleaned']



all_text = pd.concat([train_text, test_text])
word_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    #analyzer='word',

    token_pattern=r'\w{1,}',

    stop_words='english',

    ngram_range=(1, 1),

    max_features=10000)

word_vectorizer.fit(all_text)

train_word_features = word_vectorizer.transform(train_text)

test_word_features = word_vectorizer.transform(test_text)



char_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    #analyzer='char',

    stop_words='english',

    ngram_range=(2, 6),

    max_features=50000)

char_vectorizer.fit(all_text)

train_char_features = char_vectorizer.transform(train_text)

test_char_features = char_vectorizer.transform(test_text)
train_features = hstack([train_char_features, train_word_features])

test_features = hstack([test_char_features, test_word_features])



scores = []

submission = pd.DataFrame.from_dict({'id': test.index})

for class_name in class_names:

    train_target = train[class_name]

    classifier = RandomForestClassifier(max_leaf_nodes=100,n_estimators=200, random_state=10)

    #classifier = LogisticRegression(C=0.1, solver='sag')

    #classifier = DecisionTreeRegressor(max_leaf_nodes=250, random_state=0)

    #classifier = xgb.XGBClassifier(random_state=10,learning_rate=0.01)

    



    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))

    scores.append(cv_score)

    print('CV score for class {} is {}'.format(class_name, cv_score))



    classifier.fit(train_features, train_target)

    submission[class_name] = classifier.predict(test_features)
print('Total CV score is {}'.format(np.mean(scores)))



submission.to_csv('result_to_submission.csv', index=False)
#https://www.kaggle.com/vanshjatana/a-simple-guide-to-text-cleaning

    