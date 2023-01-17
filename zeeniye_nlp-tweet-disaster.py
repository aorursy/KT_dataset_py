import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')
sns.set_style('whitegrid')
tweet_train = pd.read_csv('../input/nlp-getting-started/train.csv')
tweet_test = pd.read_csv('../input/nlp-getting-started/test.csv')

len(tweet_train), len(tweet_test)
tweet_train.head()
tweet_train.info()
tweet_test.head()
tweet_test.info()
tweet_train.isnull().sum()
tweet_test.isnull().sum()
plt.plot([61/7613 *100, 2533/7613 *100, 26/3263 *100, 1105/3263 *100])
sns.countplot(tweet_train['target'])
tweet_train['keyword'] = tweet_train['keyword'].str.replace('%20', '_')
tweet_test['keyword'] = tweet_test['keyword'].str.replace('%20', '_')
tweet_train['length'] = tweet_train['text'].apply(len)
tweet_test['length'] = tweet_test['text'].apply(len)
tweet_train['location'].fillna('-',inplace=True)
tweet_test['location'].fillna('-',inplace=True)
tweet_train['location_known'] = tweet_train['location'].notnull()
tweet_test['location_known'] = tweet_test['location'].notnull()
tweet_train.drop('location', axis=1, inplace=True)
tweet_test.drop('location', axis=1, inplace=True)
tweet_train['keyword'].fillna('no_key',inplace=True)
tweet_test['keyword'].fillna('no_key',inplace=True)
import string
from nltk.corpus import stopwords
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
key_train = pd.get_dummies(tweet_train['keyword'], drop_first=True)
key_test = pd.get_dummies(tweet_test['keyword'], drop_first=True)
tweet_train.drop('keyword', axis=1, inplace=True)
tweet_test.drop('keyword', axis=1, inplace=True)
tweet_train = pd.concat([tweet_train,key_train], axis=1)
tweet_test = pd.concat([tweet_test,key_test], axis=1)
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
X = tweet_train.drop(['id','target'], axis=1)
y = tweet_train['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
X_train.reset_index(inplace=True, drop=True)
X_test.reset_index(inplace=True, drop=True)
y_train.reset_index(inplace=True, drop=True)
y_test.reset_index(inplace=True, drop=True)
cvz = CountVectorizer(stop_words='english')
tfid = TfidfTransformer()
pl = Pipeline([
    ('vectorizer', cvz),
    ('tfid', tfid)
])
text_X_train = pd.DataFrame(cvz.fit_transform(X_train['text']).todense(), columns=cvz.get_feature_names())
text_X_test = pd.DataFrame(cvz.transform(X_test['text']).todense(), columns=cvz.get_feature_names())
txt_train = pd.DataFrame(tfid.fit_transform(text_X_train).todense())
txt_test = pd.DataFrame(tfid.transform(text_X_test).todense())
txt_test.shape, txt_train.shape, X_train.shape, X_test.shape
X_train.drop('text', axis=1, inplace=True)
X_test.drop('text', axis=1, inplace=True)
X_train = pd.concat([X_train, text_X_train], axis=1, ignore_index=True)
X_test = pd.concat([X_test, text_X_test], axis=1, ignore_index=True)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
nb = MultinomialNB()
nb.fit(X_train, y_train)
logR = LogisticRegression(max_iter=1000)
logR.fit(X_train, y_train)
lgbC = LGBMClassifier()
lgbC.fit(X_train, y_train)
rf = RandomForestClassifier()
nb.score(X_test, y_test)
logR.score(X_test, y_test)
lgbC.score(X_test, y_test)
pred = nb.predict(X_test)
logpred = logR.predict(X_test)
lgbpred = lgbC.predict(X_test)
print(confusion_matrix(y_test, pred),'\n')
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, logpred),'\n')
print(classification_report(y_test, logpred))
print(confusion_matrix(y_test, lgbpred),'\n')
print(classification_report(y_test, lgbpred))
text_X_full_train = pd.DataFrame(cvz.fit_transform(X['text']).todense(), columns=cvz.get_feature_names())
txt_full_train = pd.DataFrame(tfid.fit_transform(text_X_full_train).todense())
txt_full_train.shape
X.drop('text', axis=1, inplace=True)
X = pd.concat([X, text_X_full_train],axis=1)
logR.fit(X, y)
X_predict = tweet_test.drop('id', axis=1)
text_X_real = pd.DataFrame(cvz.transform(X_predict['text']).todense(), columns=cvz.get_feature_names())
txt_real = pd.DataFrame(tfid.transform(text_X_real).todense())
X_predict.drop('text', axis=1, inplace=True)
X_predict = pd.concat([X_predict,text_X_real], axis=1)
y_pred = logR.predict(X_predict)
output_data = pd.DataFrame()
output_data['id'] = tweet_test['id']
output_data['target'] = y_pred
output_data.tail()
output_data.to_csv('tweet_disaster_nlp_pred.csv', index=False)