# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import warnings

warnings.simplefilter('ignore')
df_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

tweet=df_train[['text','target']]

print(tweet)

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')



sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
def remove_punctuation(text):

    import string

    translator =str.maketrans('','',string.punctuation)

    return text.translate(translator)

tweet['text'] = tweet.loc[:,('text')].apply(remove_punctuation)



print(tweet.head())
from nltk.corpus import stopwords 

Stopword = set(stopwords.words('english'))

print(Stopword)
def remove_stopword(text):

    text = [word.lower() for word in text.split() if word.lower() not in Stopword]

    return " ".join(text)
tweet['text']=tweet.text.apply(remove_stopword)

print(tweet.head())
from nltk.stem.snowball import SnowballStemmer



stemmer = SnowballStemmer("english")

def stemming_start(text):

    

    stopwords = [stemmer.stem(word) for word in text.split()]

    return " ".join(stopwords) 
tweet['text'] = tweet.text.apply(stemming_start)

print(tweet.head())
from sklearn.feature_extraction.text import TfidfVectorizer



vectorize = TfidfVectorizer()

vectorize.fit(tweet.text)

#print(vectorize.vocabulary_)

print(vectorize.idf_)

vector = vectorize.transform(tweet.text)

print(vector.shape)

print(type(vector))

print(vector.toarray())

X =vectorize.transform(tweet.text).toarray()

y =tweet['target'].values

print (X.shape,y.shape)
from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import train_test_split



from sklearn.metrics import f1_score
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)



logic = LogisticRegression()



logic.fit(X_train,y_train)



predicted = logic.predict(X_test)



score = f1_score(predicted,y_test)

print(score)
tweets_test = test['text']

test_X = vectorize.transform(tweets_test).todense()

test_X.shape

lr_pred = logic.predict(test_X)

print(lr_pred)

sub['target'] = lr_pred

sub.to_csv("submission.csv", index=False)

sub.head()
