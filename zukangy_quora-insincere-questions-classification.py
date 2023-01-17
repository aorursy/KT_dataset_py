!pip install nltk
import pandas as pd
import numpy as np
import nltk
import string
from nltk.tokenize import WordPunctTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print('train shape:', train.shape)
print('\nPercentage of insincere questions in the training set:\n', len(train[train.target == 1])/len(train))
print('\ntest shape:', test.shape)
def prediction_tfidf(train):
    tfidf = TfidfVectorizer(ngram_range=(1,4), stop_words = 'english', analyzer = 'word', 
                            smooth_idf = True, sublinear_tf = True)
    logit = LogisticRegression(solver = 'sag')
    
    X = train.question_text
    y = train.target
    
    #split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    
    X_train_matrix = tfidf.fit_transform(X_train)
    X_test_matrix = tfidf.transform(X_test)
    
    logit.fit(X_train_matrix, y_train)
    pred_logit = logit.predict(X_test_matrix)
    
    accuracy_logit = accuracy_score(y_test, pred_logit)
    recall_logit = recall_score(y_test, pred_logit)
    precision_logit = precision_score(y_test, pred_logit)
    f1_logit = f1_score(y_test, pred_logit)
    
    df = pd.DataFrame({'logistic': [accuracy_logit, recall_logit, precision_logit, f1_logit]})
    df.index = ['accuracy', 'recall', 'precision', 'F1']
    
    return df
#prediction_tfidf(train)
def tokenize(text):
    '''Take out the tokens starting with a digit or a punctuation'''
    text = WordPunctTokenizer().tokenize(text)
    clean_text = []
    
    for i in text:
        if i[0] not in (string.punctuation + '0123456789'):
            clean_text.append(i)
            
    return clean_text
vec = TfidfVectorizer(ngram_range=(1,4), stop_words = 'english', analyzer = 'word', 
                            smooth_idf = True, sublinear_tf = True, tokenizer = tokenize)
logit = LogisticRegression(solver = 'sag', max_iter = 250, class_weight = 'balanced', C = 0.5)

X_train = vec.fit_transform(train.question_text)
y_train = train.target
X_test = vec.transform(test.question_text)

logit.fit(X_train, y_train)
predict = logit.predict(X_test)

submission_df = pd.DataFrame({'qid': test.qid, 'prediction': predict})
submission_df.to_csv('submission.csv', index = False)