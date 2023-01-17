import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df=pd.read_csv('../input/nlp-getting-started/train.csv')

test=pd.read_csv('../input/nlp-getting-started/test.csv')

df.head()
test.head()
df.info()
df.isnull().sum()
df.shape
df.head()
df['target'].value_counts()
df['keyword'].nunique()
df['text']
df_len=df['text'].str.len()

test_len=test['text'].str.len()

plt.hist(df_len,bins=20,label='train_text')

plt.hist(test_len,bins=20,label='test_text')
from wordcloud import WordCloud
df_text=' '.join([x for x in df['text']])
cloud=WordCloud(width=400,height=300).generate(df_text)

plt.figure(figsize=(10,10))

plt.imshow(cloud,interpolation='bilinear')
df.head()
df=df.drop('id',1)

test=test.drop('id',1)
df['keyword'].mode()
test['keyword'].mode()
df['keyword']=df['keyword'].fillna('fatalities') 

test['keyword']=test['keyword'].fillna('deluged') 
df['location']=df['location'].fillna('999') 

test['location']=test['location'].fillna('999') 
df.isnull().sum()
#removing numbers and puncuations

df['text']=df['text'].str.replace('[^a-zA-Z@#]',' ')

test['text']=test['text'].str.replace('[^a-zA-Z@#]',' ')
df['keyword']=df['keyword'].str.replace('[^a-zA-Z@#]',' ')

test['keyword']=test['keyword'].str.replace('[^a-zA-Z@#]',' ')
df['location']=df['location'].str.replace('[^a-zA-Z@#]',' ')

test['location']=test['location'].str.replace('[^a-zA-Z@#]',' ')
#removing smaller words

df['text']=df['text'].apply(lambda x:' '.join([word for word in x.split() if len(word)>3]))

test['text']=test['text'].apply(lambda x:' '.join([word for word in x.split() if len(word)>3]))
#making all words lowercase

df['text']=df['text'].apply(lambda x:x.lower())

test['text']=test['text'].apply(lambda x:x.lower())
df.head()
from sklearn.feature_extraction.text import TfidfVectorizer

tf=TfidfVectorizer(stop_words='english',max_features=1000)

tf.fit(df['text'])

df_tf=pd.DataFrame(tf.transform(df['text']).toarray(),columns=tf.get_feature_names()).add_prefix('tfidf_')

df=df.join(df_tf)

df=df.drop('text',1)
test_tf=pd.DataFrame(tf.transform(test['text']).toarray(),columns=tf.get_feature_names()).add_prefix('tfidf_')

test=test.join(test_tf)

test=test.drop('text',1)
from sklearn.feature_extraction.text import TfidfVectorizer

tf=TfidfVectorizer(stop_words='english',max_features=200)

tf.fit(df['keyword'])

df_tf=pd.DataFrame(tf.transform(df['keyword']).toarray(),columns=tf.get_feature_names()).add_prefix('tfidf1_')

df=df.join(df_tf)

df=df.drop('keyword',1)
test_tf=pd.DataFrame(tf.transform(test['keyword']).toarray(),columns=tf.get_feature_names()).add_prefix('tfidf1_')

test=test.join(test_tf)

test=test.drop('keyword',1)
from sklearn.feature_extraction.text import TfidfVectorizer

tf=TfidfVectorizer(stop_words='english',max_features=500)

tf.fit(df['location'])

df_tf=pd.DataFrame(tf.transform(df['location']).toarray(),columns=tf.get_feature_names()).add_prefix('tfidf2_')

df=df.join(df_tf)

df=df.drop('location',1)
test_tf=pd.DataFrame(tf.transform(test['location']).toarray(),columns=tf.get_feature_names()).add_prefix('tfidf2_')

test=test.join(test_tf)

test=test.drop('location',1)
df.shape
test.shape
df.isnull().sum()
test.isnull().sum()
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC
x=df.drop('target',1)

y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,stratify=y)
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(max_depth=9)

dt.fit(x_train,y_train)

dt.score(x_test,y_test)
from sklearn.linear_model import LogisticRegression

log=LogisticRegression()

log.fit(x_train,y_train)

y_log=log.predict(x_test)

f1_score(y_test,y_log)
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=70,n_jobs=-1)

rf.fit(x_train,y_train)

y_rf=rf.predict(x_test)

f1_score(y_test,y_rf)
from sklearn.ensemble import BaggingClassifier

gab=BaggingClassifier(n_estimators=400)

gab.fit(x_train,y_train)

y_gab=gab.predict(x_test)

f1_score(y_test,y_gab)
from xgboost import XGBClassifier

clf= XGBClassifier(n_estimators=5,max_depth=9)#,subsample=0.7,colsample_bytree=0.7)

clf.fit(x,y)

y_clf=clf.predict(x_test)

f1_score(y_test,y_clf,average='weighted')
from catboost import CatBoostClassifier

model=CatBoostClassifier(iterations=1, depth=9, learning_rate=0.3, loss_function='Logloss',

                         eval_metric='Accuracy')

model.fit(x_train,y_train)
y_test=gab.predict(test)
sub=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
sub['target']=y_test
sub['target'].value_counts()
sub.to_csv('answer.csv',index=False)
sub