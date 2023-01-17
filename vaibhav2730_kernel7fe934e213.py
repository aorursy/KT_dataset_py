# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import string
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#load_data
train_df= pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df= pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

print(train_df.shape)
print(train_df.head(5))
#number of non-null values in each column
print(train_df.count())
#number of unique values in each column
train_df.nunique()
train_df['keyword'].unique()
train_df['location'].unique()
print(train_df[train_df['target']==1]['location'].nunique())
print(train_df[train_df['target']==0]['location'].nunique())
print(train_df[train_df['target']==1]['keyword'].nunique())
print(train_df[train_df['target']==0]['keyword'].nunique())
y_features=train_df['target']
train_df=train_df.drop('target',axis=1)
#one_hot_encoding of keyword and location
one_hot= pd.get_dummies(train_df['keyword'])
train_df=train_df.drop('keyword', axis=1)
train_df=train_df.join(one_hot)
one_hot= pd.get_dummies(train_df['location'])
train_df=train_df.drop('location', axis=1)
train_df=train_df.join(one_hot)

#test set
one_hot= pd.get_dummies(test_df['keyword'])
test_df=test_df.drop('keyword', axis=1)
test_df=test_df.join(one_hot)
one_hot= pd.get_dummies(test_df['location'])
test_df=test_df.drop('location', axis=1)
test_df=test_df.join(one_hot)

train_df, test_df = train_df.align(test_df, join="inner", axis=1)
train_df
#introducing two new features- text_len
# train_df['text_len']=[len(x) for x in train_df['text']]
train_df['text_len']=train_df['text'].apply(lambda x: len(x)- x.count(" "))
test_df['text_len']=test_df['text'].apply(lambda x: len(x)- x.count(" "))

train_df.head(5)
#introducing punctuation_len as new feature
import string
def count_punct(text):
    count= sum([1 for char in text if char in string.punctuation])
    return count

train_df['punct']= train_df['text'].apply(lambda x: count_punct(x))
test_df['punct']= test_df['text'].apply(lambda x: count_punct(x))
train_df.head(5)
#we need to convert text column to vectors
import nltk
import string

#define a function to clean text and tokenize and stem words
def clean_text(text):
    stopwords= nltk.corpus.stopwords.words('english')
    ps=nltk.PorterStemmer()
    text="".join([word.lower() for word in text if word not in string.punctuation])
    tokens= nltk.tokenize.word_tokenize(text)
    text=[ps.stem(word) for word in tokens if word not in stopwords]
    return text
    
# print(clean_text(train_df.iloc[0]['text']))

#apply tf-idf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer(analyzer=clean_text)
X_tfidf_vect=tf.fit(train_df['text'])
X_tfidf= X_tfidf_vect.transform(train_df['text'])
X_tfidf_test=X_tfidf_vect.transform(test_df['text'])
print(X_tfidf.shape)
print(tf.get_feature_names())


    
    
#change the sparse matrices to dataframe
X_tfidf_df=pd.DataFrame(X_tfidf.toarray())
X_tfidf_df.columns= tf.get_feature_names()
X_tfidf_df
#change the sparse matrices to dataframe
X_tfidf_test_df=pd.DataFrame(X_tfidf_test.toarray())
X_tfidf_test_df.columns= tf.get_feature_names()
X_tfidf_test_df
# #let us evaluate the features we created- text_len, punct
# import matplotlib.pyplot as plt
# import numpy as np
# %matplotlib inline

# bins=np.linspace(0,300,50)
# plt.hist(train_df[train_df['target']==1]['text_len'], bins, alpha=0.5, density= True, label='1')
# plt.hist(train_df[train_df['target']==0]['text_len'], bins, alpha=0.5, density= True, label='0')
# plt.legend('upper right')
# plt.show()

# #not very much difference in both. Not very useful feature
# let us use gradient boosting for this classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split


def train_GB(n_est, max_depth, lr):
    gb = GradientBoostingClassifier(n_estimators=n_est, max_depth= max_depth, learning_rate=lr)
    gb_model= gb.fit(X_train, y_train)
    return gb_model
    
    

train_df=train_df.drop(['text'], axis=1)
X_features= pd.concat([train_df,X_tfidf_df], axis=1)

test_df=test_df.drop(['text'],axis=1)
X_test_features= pd.concat([test_df,X_tfidf_test_df], axis=1)
X_train, X_test,y_train, y_test = train_test_split(X_features, y_features, test_size=0.2)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

y_train.head(5)
model= train_GB(200,5,0.1)
y_pred= model.predict(X_test)
precision, recall, fscore, support = score(y_test, y_pred, pos_label=1,average='binary')
print("n_est {} , Max_depth {}, learning rate {}, Precision {}, Recall {}, Fscore {}  Accuracy {}".format(150,3,0.1,round(precision,3),round(recall,3),round(fscore,3),round((y_pred==y_test).sum()/len(y_pred),3) ))
    
#submission
submission_df=pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
submission_df
test_df
test_pred=model.predict(X_test_features)

test_pred
submission_df['target']=test_pred
submission_df.to_csv("submission.csv", index=False, header=True)
