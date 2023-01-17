import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
dataset = pd.read_csv(r'../input/Tweets.csv')
dataset.head()
(len(dataset) - dataset.count())/len(dataset)
dataset = dataset.drop(['airline_sentiment_gold','negativereason_gold','tweet_coord'],axis=1)
dataset.head(3)
mood_count=dataset['airline_sentiment'].value_counts()
mood_count
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='airline_sentiment',data=dataset,order=['negative','neutral','positive'])
plt.show()
sns.factorplot(x = 'airline_sentiment',data=dataset,
               order = ['negative','neutral','positive'],kind = 'count',col_wrap=3,col='airline',size=4,aspect=0.6,sharex=False,sharey=False)
plt.show()
dataset['negativereason'].value_counts()
sns.factorplot(x = 'airline',data = dataset,kind = 'count',hue='negativereason',size=12,aspect=.9)
plt.show()
sns.factorplot(x = 'negativereason',data=dataset,kind='count',col='airline',size=9,aspect=.8,col_wrap=2,sharex=False,sharey=False)
plt.show()
import re
import nltk
import time
start_time = time.time()
#remove words which are starts with @ symbols
dataset['text'] = dataset['text'].map(lambda x:re.sub('@\w*','',str(x)))
#remove special characters except [a-zA-Z]
dataset['text'] = dataset['text'].map(lambda x:re.sub('[^a-zA-Z]',' ',str(x)))
#remove link starts with https
dataset['text'] = dataset['text'].map(lambda x:re.sub('http.*','',str(x)))
end_time = time.time()
#total time consume to filter data
end_time-start_time
dataset['text'].head()
dataset['text'] = dataset['text'].map(lambda x:str(x).lower())
dataset['text'].head(2)
from nltk.corpus import stopwords
corpus = []
none=dataset['text'].map(lambda x:corpus.append(' '.join([word for word in str(x).strip().split() if not word in set(stopwords.words('english'))])))                                     
corpus[:4]
X = pd.DataFrame(data=corpus,columns=['comment_text'])
X.head()
y = dataset['airline_sentiment'].map({'neutral':1,'negative':-1,'positive':1})
y.head(2)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
from sklearn.feature_extraction.text import TfidfVectorizer
vector = TfidfVectorizer(stop_words='english',sublinear_tf=True,strip_accents='unicode',analyzer='word',token_pattern=r'\w{2,}',ngram_range=(1,1),max_features=30000)
#token_patten #2 for word length greater than 2>=
X_train_word_feature = vector.fit_transform(X_train['comment_text']).toarray()
X_test_word_feature = vector.transform(X_test['comment_text']).toarray()
print(X_train_word_feature.shape,X_test_word_feature.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
classifier = LogisticRegression()
classifier.fit(X_train_word_feature,y_train)
y_pred = classifier.predict(X_test_word_feature)
cm = confusion_matrix(y_test,y_pred)
acc_score = accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred),'\n',cm,'\n',acc_score)
y_pred_prob = classifier.predict_proba(X_train_word_feature)
y_pred_prob[:5]