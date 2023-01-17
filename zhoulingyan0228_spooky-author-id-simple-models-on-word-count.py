import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.layers import *
from keras.losses import *
from keras.optimizers import *
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
import seaborn as sns
import re

df = pd.read_csv("../input/train.csv")
df.head()
labelEncoder = LabelEncoder().fit(df['author'])
y = labelEncoder.transform(df['author'])
tokenizer = Tokenizer(lower=False)
tokenizer.fit_on_texts(df['text'])
def get_model(name):
    if name=='LogisticRegression':
        return LogisticRegression()
    elif name=='DecisionTreeClassifier':
        return DecisionTreeClassifier()
    elif name=='RandomForestClassifier':
        return RandomForestClassifier()
    elif name=='AdaBoostClassifier':
        return AdaBoostClassifier(n_estimators=20)
    elif name=='MultinomialNB':
        return MultinomialNB()
metrics_df = pd.DataFrame(columns=['model', 'tokenizer_mode', 'metrics', 'value'])
for model_name in ['LogisticRegression']:
    for mode in ['binary', 'count', 'tfidf']:
        print('=============', model_name, ' - ', mode, '=============')
        model = get_model(model_name)
        X = tokenizer.texts_to_matrix(df['text'], mode=mode)
        kfold = KFold(4)
        losses=[]
        accuracys=[]
        for train_index, test_index in kfold.split(X):
            model.fit(X[train_index], y[train_index])
            yp_prob = model.predict_proba(X[test_index])
            yp_class = np.argmax(yp_prob, axis=1)
            losses.append(log_loss(y[test_index], yp_prob))
            accuracys.append(accuracy_score(y[test_index], yp_class))
        print ('avg log_loss=', np.mean(np.array(losses)), )
        print ('avg accuracy_score=', np.mean(np.array(accuracys)))
        print ('std log_loss=', np.std(np.array(losses)), )
        print ('std accuracy_score=', np.std(np.array(accuracys)))
        metrics_df = metrics_df.append({'model':model_name, 'tokenizer_mode':mode, 'metrics':'loss', 'value':np.mean(np.array(losses))}, ignore_index=True)
        metrics_df = metrics_df.append({'model':model_name, 'tokenizer_mode':mode, 'metrics':'accuracy', 'value':np.mean(np.array(accuracys))}, ignore_index=True)

sns.factorplot(data=metrics_df, x='model', y='value', col='metrics', hue='tokenizer_mode', kind="bar", ci=None)
metrics_df = pd.DataFrame(columns=['model', 'tokenizer_mode', 'metrics', 'value'])
    
for model_name in ['LogisticRegression', 'MultinomialNB', 'DecisionTreeClassifier', 'RandomForestClassifier', 'AdaBoostClassifier']:
    for mode in ['count']:
        print('=============', model_name, ' - ', mode, '=============')
        model = get_model(model_name)
        X = tokenizer.texts_to_matrix(df['text'], mode=mode)
        kfold = KFold(4)
        losses=[]
        accuracys=[]
        for train_index, test_index in kfold.split(X):
            model.fit(X[train_index], y[train_index])
            yp_prob = model.predict_proba(X[test_index])
            yp_class = np.argmax(yp_prob, axis=1)
            losses.append(log_loss(y[test_index], yp_prob))
            accuracys.append(accuracy_score(y[test_index], yp_class))
        print ('avg log_loss=', np.mean(np.array(losses)), )
        print ('avg accuracy_score=', np.mean(np.array(accuracys)))
        print ('std log_loss=', np.std(np.array(losses)), )
        print ('std accuracy_score=', np.std(np.array(accuracys)))
        metrics_df = metrics_df.append({'model':model_name, 'tokenizer_mode':mode, 'metrics':'loss', 'value':np.mean(np.array(losses))}, ignore_index=True)
        metrics_df = metrics_df.append({'model':model_name, 'tokenizer_mode':mode, 'metrics':'accuracy', 'value':np.mean(np.array(accuracys))}, ignore_index=True)
        
g=sns.factorplot(data=metrics_df, x='model', y='value', col='metrics', hue='tokenizer_mode', kind="bar", ci=None, sharey=False)
g.set_xticklabels(rotation=30)
def generate_ngram(doc, n=2):
    words = re.sub(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^_\'`{|}~]','',doc).split()
    length = len(words)
    for i in range(2, n+1):
        for j in range(0, length - i + 1):
            ngram = words[j]
            for k in range(1, i):
                ngram += '_'+words[j+k]
            words.append(ngram)
    return ' '.join(words)
df['text'] = df['text'].apply(generate_ngram)
tokenizer = Tokenizer(lower=False)
tokenizer.fit_on_texts(df['text'])

metrics_df = pd.DataFrame(columns=['model', 'tokenizer_mode', 'metrics', 'value'])
for model_name in ['LogisticRegression']:
    for mode in ['count']:
        print('=============', model_name, ' - ', mode, '=============')
        model = get_model(model_name)
        X = tokenizer.texts_to_matrix(df['text'], mode=mode)
        kfold = KFold(4)
        losses=[]
        accuracys=[]
        for train_index, test_index in kfold.split(X):
            model.fit(X[train_index], y[train_index])
            yp_prob = model.predict_proba(X[test_index])
            yp_class = np.argmax(yp_prob, axis=1)
            losses.append(log_loss(y[test_index], yp_prob))
            accuracys.append(accuracy_score(y[test_index], yp_class))
        print ('avg log_loss=', np.mean(np.array(losses)), )
        print ('avg accuracy_score=', np.mean(np.array(accuracys)))
        print ('std log_loss=', np.std(np.array(losses)), )
        print ('std accuracy_score=', np.std(np.array(accuracys)))
        metrics_df = metrics_df.append({'model':model_name, 'tokenizer_mode':mode, 'metrics':'loss', 'value':np.mean(np.array(losses))}, ignore_index=True)
        metrics_df = metrics_df.append({'model':model_name, 'tokenizer_mode':mode, 'metrics':'accuracy', 'value':np.mean(np.array(accuracys))}, ignore_index=True)

g=sns.factorplot(data=metrics_df, x='model', y='value', col='metrics', hue='tokenizer_mode', kind="bar", ci=None, sharey=False)
g.set_xticklabels(rotation=30)
test_df = pd.read_csv("../input/test.csv")
tokenizer = Tokenizer(lower=False)
tokenizer.fit_on_texts(df['text'])
model = get_model('LogisticRegression')
X = tokenizer.texts_to_matrix(df['text'], mode='count')
model.fit(X, y)
X_test = tokenizer.texts_to_matrix(test_df['text'], mode='count')
yp_prob = model.predict_proba(X_test)
submission_df = pd.concat([test_df[['id']], pd.DataFrame(yp_prob, columns=labelEncoder.classes_)], axis=1)
submission_df.to_csv('submission.csv', index=False)