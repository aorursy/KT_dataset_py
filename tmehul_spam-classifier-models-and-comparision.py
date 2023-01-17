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
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/spam.csv',encoding='latin-1' )
df.drop(df.columns[[2,3,4]], axis = 1, inplace = True) 
df['length'] = df['v2'].apply(len)
df.head()
len(df)
df['v1'].value_counts()
### Balance the Data
ham = df[df['v1'] == 'ham']
ham.head()
spam = df[df['v1'] == 'spam']
spam.head()
ham.shape ,spam.shape
ham = ham.sample(spam.shape[0])
ham.shape
data = ham.append(spam,ignore_index=True)
data.head()
data.head()
### Exploratory Data Analysis
plt.figure()
plt.hist(data[data['v1'] == 'ham']['length'] , bins = 100 ,alpha = 0.7)
plt.hist(data[data['v1'] == 'spam']['length'] , bins = 100 ,alpha = 0.7)
plt.show()
### Data Prepration
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
X_train,X_test,y_train,y_test = train_test_split(data['v2'] , data['v1'],test_size = 0.3 ,random_state = 0 ,shuffle = True ,stratify = data['v1'])
### Bag of word (unique) Creation
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
### Pipeline and RF
clf = Pipeline([('tfidf' , TfidfVectorizer()) , ('clf',RandomForestClassifier(n_estimators=100 , n_jobs=-1))])
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
RF_acc_score = accuracy_score(y_test,y_pred)
RF_acc_score
clf.predict(["Hey,you won free tickets this summer.Text 'WON' @ 75489.....Hurry few Hours Left."])
# Pipeline and  SVC
clf = Pipeline([('tfidf' , TfidfVectorizer()) , ('clf',SVC(C=2000,gamma='auto',))])
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
SCV_acc_score = accuracy_score(y_test,y_pred)
SCV_acc_score
clf.predict(["wassup??"])
clf.predict(["Are you free Tonight?? call me @327001 now and throw away your boredom."])
