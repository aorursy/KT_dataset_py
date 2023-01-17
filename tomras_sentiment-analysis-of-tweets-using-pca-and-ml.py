import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
df = pd.read_csv("../input/Tweets.csv")
df.head()
df.dtypes
df.isnull().sum()
df = df.drop(['negativereason', 
              'negativereason_confidence', 
              'airline_sentiment_gold', 
              'negativereason_gold', 
              'tweet_coord', 
              'tweet_created', 
              'tweet_location', 
              'user_timezone', 
              'tweet_id',
              'name',
              'airline_sentiment_confidence',
              'retweet_count'], axis=1)

df.head()
df['airline_sentiment'].value_counts().plot(kind='bar')
airline_categorical = pd.get_dummies(df['airline'])
# df = df.drop(['airline'], axis=1)
df = pd.concat([df, airline_categorical], axis=1)
df.head()
stop_words = set(stopwords.words('english'))

df['text'] = df['text'].apply(lambda x: re.sub('[^a-z]', ' ', x.lower()))
df['text'] = df['text'].apply(lambda x: re.sub(' +', ' ', x))
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords.words('english')]))

df.head()
df['target'] = df['airline_sentiment'].apply(lambda x: 0 if x == 'negative' else 1 if x == 'neutral' else 2)
df = df.drop(['airline_sentiment'], axis=1)
df.head()
df = df.drop_duplicates()
df = df.reset_index(drop=True)
df.describe(include='all')
df_train, df_test = train_test_split(df, test_size=0.3)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
print(df_train.shape, df_test.shape)
df_train.head()
vectorizer = TfidfVectorizer()
text_features_train = vectorizer.fit_transform(df_train['text'])
text_features_train.shape
features_train = np.concatenate([text_features_train.toarray(), df_train[['American', 
                                                                          'Delta', 
                                                                          'Southwest', 
                                                                          'US Airways', 
                                                                          'United', 
                                                                          'Virgin America']].values], axis=1)
features_train.shape
pca = PCA(n_components=2)
features_train = pca.fit_transform(features_train)
features_train.shape
df_features_train = pd.DataFrame(features_train)
df_features_train = pd.concat([df_features_train, df_train[['target']]], axis=1, ignore_index=True)
df_features_train.columns = ['pca_1', 'pca_2', 'target']
df_features_train.describe(include='all')
cmap = {0: 'red', 1: 'blue', 2: 'green'}
df_features_train.plot(kind='scatter', x='pca_1', y='pca_2', c=[cmap.get(t, 'black') for t in df_features_train['target']])
Classifiers = [
    KNeighborsClassifier(3),
    KNeighborsClassifier(5),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200),
    AdaBoostClassifier(),
    GaussianNB()]
text_features_test = vectorizer.transform(df_test['text'])
features_test = np.concatenate([text_features_test.toarray(), df_test[['American', 
                                                                       'Delta', 
                                                                       'Southwest', 
                                                                       'US Airways', 
                                                                       'United', 
                                                                       'Virgin America']].values], axis=1)
features_test = pca.transform(features_test)
df_features_test = pd.DataFrame(features_test)
df_features_test = pd.concat([df_features_test, df_test[['target']]], axis=1, ignore_index=True)
df_features_test.columns = ['pca_1', 'pca_2', 'target']
df_features_test.shape
for c in Classifiers:
    fit = c.fit(df_features_train[['pca_1', 'pca_2']], df_features_train[['target']])
    pred = fit.predict(df_features_test[['pca_1', 'pca_2']])

    accuracy = accuracy_score(pred, df_features_test[['target']])

    print('Accuracy of ' + c.__class__.__name__ + 'is ' + str(accuracy))  
df_train['airline'] = df_train['airline'].apply(lambda x: 1 if x == 'American' else 2 if x == 'Delta' else 3 if x =='Southwest' else 4 if x == 'US Airways' else 5 if x == 'United' else 6 if x == 'Virgin America' else 0)
df_test['airline'] = df_test['airline'].apply(lambda x: 1 if x == 'American' else 2 if x == 'Delta' else 3 if x =='Southwest' else 4 if x == 'US Airways' else 5 if x == 'United' else 6 if x == 'Virgin America' else 0)

vectorizer = TfidfVectorizer()
text_features_train = vectorizer.fit_transform(df_train['text'])

features_train = np.concatenate([text_features_train.toarray(), df_train[['airline']].values], axis=1)

pca = PCA(n_components=2)
features_train = pca.fit_transform(features_train)

df_features_train = pd.DataFrame(features_train)
df_features_train = pd.concat([df_features_train, df_train[['target']]], axis=1, ignore_index=True)
df_features_train.columns = ['pca_1', 'pca_2', 'target']
df_features_train.describe(include='all')

text_features_test = vectorizer.transform(df_test['text'])
features_test = np.concatenate([text_features_test.toarray(), df_test[[ 'airline']].values], axis=1)

features_test = pca.transform(features_test)
df_features_test = pd.DataFrame(features_test)
df_features_test = pd.concat([df_features_test, df_test[['target']]], axis=1, ignore_index=True)
df_features_test.columns = ['pca_1', 'pca_2', 'target']
for c in Classifiers:
    fit = c.fit(df_features_train[['pca_1', 'pca_2']], df_features_train[['target']])
    pred = fit.predict(df_features_test[['pca_1', 'pca_2']])

    accuracy = accuracy_score(pred, df_features_test[['target']])

    print('Accuracy of ' + c.__class__.__name__ + 'is ' + str(accuracy)) 