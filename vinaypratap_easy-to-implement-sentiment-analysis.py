import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

import re

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk import word_tokenize

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.svm import LinearSVC

from sklearn.metrics import f1_score, accuracy_score

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

pd.set_option('display.notebook_repr_html', True)
train = pd.read_csv("../input/train.csv")

train.head()
test = pd.read_csv("../input/test.csv")

test.head()
df = pd.concat([train,test])

df.head()
train.shape, test.shape, df.shape
df.dtypes
df.isnull().sum()    # label is expected one as test dataset does not have label
df['label'].value_counts()
sns.countplot(df['label'])
# Cleaning Raw tweets

def clean_text(text):

    

    #remove emails

    text = ' '.join([i for i in text.split() if '@' not in i])

    

    #remove web address

    text = re.sub('http[s]?://\S+', '', text)

    

    #Filter to allow only alphabets

    text = re.sub(r'[^a-zA-Z\']', ' ', text)

    

    #Remove Unicode characters

    text = re.sub(r'[^\x00-\x7F]+', '', text)

    

    #Convert to lowercase to maintain consistency

    text = text.lower()

    

    #remove double spaces 

    text = re.sub('\s+', ' ',text)

    

    return text



df["clean_tweet"] = df['tweet'].apply(lambda x: clean_text(x))
df.head()
#defining stop words

STOP_WORDS = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'also', 'am', 'an', 'and',

              'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',

              'between', 'both', 'but', 'by', 'can', "can't", 'cannot', 'com', 'could', "couldn't", 'did',

              "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'else', 'ever',

              'few', 'for', 'from', 'further', 'get', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having',

              'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how',

              "how's", 'however', 'http', 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it',

              "it's", 'its', 'itself', 'just', 'k', "let's", 'like', 'me', 'more', 'most', "mustn't", 'my', 'myself',

              'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'otherwise', 'ought', 'our', 'ours',

              'ourselves', 'out', 'over', 'own', 'r', 'same', 'shall', "shan't", 'she', "she'd", "she'll", "she's",

              'should', "shouldn't", 'since', 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs',

              'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're",

              "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't",

              'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where',

              "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't",

              'www', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
#remove stopwords

df['cleaned_tweets'] = df['clean_tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in STOP_WORDS]))
#from nltk.stem.porter import PorterStemmer

#ps = PorterStemmer()

#df['cleaned_tweets'] = df['clean_tweet'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split() if word not in STOP_WORDS]))
#remove stopwords

#df['cleaned_tweets'] = df['clean_tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords.words('english')]))
# feature length of tweet

df['word_count'] = df['cleaned_tweets'].str.split().apply(lambda x :len(x))
# drop not needed features

df_1 = df.copy()

df.drop(['tweet','clean_tweet'],axis=1,inplace=True)

#df.drop(['tweet','clean_tweet','word_count'],axis=1,inplace=True)
df.head()
train = df[:7920]

test = df[7920:]

df_t = test.copy()
train.shape, test.shape
X = train.drop('label',axis=1)

y = train['label']
X.shape, y.shape
test.drop('label',axis=1,inplace=True)
X = X['cleaned_tweets'].astype('category')

#X = X['word_count'].astype('category')

test = test['cleaned_tweets'].astype('category')

#test = test['word_count'].astype('category')
#Train test Split

#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1)
vect = CountVectorizer(max_features = 2500)

vect.fit(X)

X_train_vec = vect.transform(X)

X_test_vec = vect.transform(test)
model = RandomForestClassifier(n_estimators=100,max_depth=5)

model.fit(X_train_vec,y)

rf = model.predict(X_test_vec)

submission = pd.DataFrame({

        "id": df_t["id"],

        "label":rf

    })



submission.to_csv('sentiments_submission.csv', index=False)
clf = SGDClassifier(max_iter=5, tol=None)

clf.fit(X_train_vec, y)

y_pred_SGD = clf.predict(X_test_vec)



submission = pd.DataFrame({

        "id": df_t["id"],

        "label":y_pred_SGD

    })



submission.to_csv('sentiments_submission_SGD.csv', index=False)
import xgboost

classifier = xgboost.XGBClassifier(n_estimators=210)

classifier.fit(X_train_vec, y)

# Predicting the Test set results

y_pred_XGB = classifier.predict(X_test_vec)

submission = pd.DataFrame({

        "id": df_t["id"],

        "label":y_pred_XGB

    })



submission.to_csv('sentiments_submission_XGB.csv', index=False)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train_vec, y)

y_pred_log_reg = clf.predict(X_test_vec)

submission = pd.DataFrame({

        "id": df_t["id"],

        "label":y_pred_log_reg

    })



submission.to_csv('sentiments_submission_LRG.csv', index=False)
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors = 4)

clf.fit(X_train_vec, y)

y_pred_knn = clf.predict(X_test_vec)

submission = pd.DataFrame({

        "id": df_t["id"],

        "label":y_pred_knn

    })



submission.to_csv('sentiments_submission_knn.csv', index=False)
from sklearn.linear_model import Perceptron

clf = Perceptron(max_iter=6, tol=None)

clf.fit(X_train_vec, y)

y_pred_perceptron = clf.predict(X_test_vec)

submission = pd.DataFrame({

        "id": df_t["id"],

        "label":y_pred_perceptron

    })



submission.to_csv('sentiments_submission_per.csv', index=False)
X_train_vec=X_train_vec.astype('float64')

X_test_vec=X_test_vec.astype('float64')
X_train_vec
import lightgbm as lgb

train_data=lgb.Dataset(X_train_vec,label=y)

#define parameters

params = {'n_estimators':213,'objective':'binary','learning_rate':0.2,'max_depth': 10,'num_leaves':'100','min_data_in_leaf':9,'max_bin':100,'boosting_type':'gbdt',}

model= lgb.train(params, train_data, 100) 

y_pred_LGB=model.predict(X_test_vec)

#rounding the values

y_pred_LGB=y_pred_LGB.round(0)

#converting from float to integer

y_pred_LGB=y_pred_LGB.astype(int)

submission = pd.DataFrame({

        "id": df_t["id"],

        "label":y_pred_LGB

    })



submission.to_csv('sentiments_submission_LGB.csv', index=False)