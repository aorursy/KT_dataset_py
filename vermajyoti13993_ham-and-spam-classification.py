import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import nltk

import string

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import warnings

warnings.filterwarnings('ignore')

data= pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',delimiter=',',encoding='latin-1')
data.head() #First five row
data.shape
data.isna().sum()
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)

data = data.rename(columns={'v1':'Labels','v2':'Message'}) 
data.info()
sns.countplot(data.Labels)

plt.title('No. of ham and spam messages')

print('{:0.2f}% of the ham massages'.format(100*(data.Labels.value_counts()[0])/len(data)))

print('{:0.2f}% of the spam massages'.format(100*(data.Labels.value_counts()[1])/len(data)))

pd.set_option('display.max_colwidth',2000)
data['message_len']=data['Message'].apply(len)
data.describe()
data.loc[data['message_len'].max()][1]
data['Text'] = data['Message'].map(lambda word :''.join([w for w in word if not w in string.punctuation]))

data['Text'] = data['Text'].map(lambda text : word_tokenize(text.lower()))

stopword = set(stopwords.words('english'))

data['Text'] = data['Text'].map(lambda token : [w for w in token if not w in stopword])
data.head()
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("english")

data['Text'] = data['Text'].map(lambda text : [stemmer.stem(w)for w in text])
data['Text'] = data['Text'].map(lambda text : ' '.join(text))
data.head()
from sklearn.feature_extraction.text import TfidfVectorizer

x = data['Text']

y = data['Labels']
tfidf = TfidfVectorizer()

tfidf_dtm = tfidf.fit_transform(x)

tfidf_data=pd.DataFrame(tfidf_dtm.toarray())

tfidf_data.head()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(tfidf_data,y,test_size=0.2)
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

nb = MultinomialNB(alpha=0.2)

nb.fit(x_train,y_train)

pred = nb.predict(x_test)

accuracy_score(pred,y_test)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

lr = LogisticRegression()

lr.fit(x_train,y_train)

pred = lr.predict(x_test)

accuracy_score(pred,y_test)