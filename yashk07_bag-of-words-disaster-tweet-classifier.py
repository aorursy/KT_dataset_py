# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('../input/nlp-getting-started/train.csv')
df.head(5)
df.info()
sns.heatmap(df.isnull())
df.drop(['location','keyword'],axis=1,inplace=True)
df
real = df[df['target']==1]
real
unreal = df[df['target']==0]
unreal
print('real disaster message percentage:',(len(real)/len(df))*100)
print('fake disaster message percentage:',(len(unreal)/len(df))*100)
sns.countplot(df['target'])
import string

string.punctuation
from nltk.corpus import stopwords
stopwords.words('english');
def message_cleaning(message):
    test_punc_removed = [char   for char in message if char not in string.punctuation]
    test_punc_removed_joined = ''.join(test_punc_removed)
    test_punc_removed_joined_clean = [word   for word in test_punc_removed_joined.split(' ') if word not in stopwords.words('english')]
    return test_punc_removed_joined_clean
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer=message_cleaning)
disaster_tweet_vectorizer = vectorizer.fit_transform(df['text'])
print(vectorizer.get_feature_names());
print(disaster_tweet_vectorizer.toarray())
disaster_tweet_vectorizer.shape
label = df['target']
label.shape
X = disaster_tweet_vectorizer
X = X.toarray()
X
y = label
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

LR = LogisticRegression()
DTC = DecisionTreeClassifier()
RFC = RandomForestClassifier()
NB = GaussianNB()
RFC.fit(X_train,y_train)
DTC.fit(X_train,y_train)
NB.fit(X_train,y_train)
LR.fit(X_train,y_train)
predict1 = RFC.predict(X_test)
predict2 = DTC.predict(X_test)
predict3 = NB.predict(X_test)
predict4 = LR.predict(X_test)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(classification_report(y_test,prediction))
print(accuracy_score(y_test,predict1))
print('\n')
print(accuracy_score(y_test,predict2))
print('\n')
print(accuracy_score(y_test,predict3))
print('\n')
print(accuracy_score(y_test,predict4))
test_df = pd.read_csv('../input/nlp-getting-started/test.csv')
test_df.head()
test_df.drop(['keyword','location'],axis=1,inplace= True)
test_df.head()
test_vectorizer = vectorizer.transform(test_df['text'])
test_vectorizer.shape
final_predictions = LR.predict(test_vectorizer)
final_predictions
submission_df = pd.DataFrame()
submission_df['id'] = test_df['id']
submission_df['target'] = final_predictions
submission_df['target'].value_counts()
