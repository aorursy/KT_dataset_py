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
tn = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv",nrows=4000)
fn = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv",nrows=4000)
fn.head()
tn['class'] = 1
fn['class'] = 0
tn.head()
df = pd.concat([tn.head(4000),fn.head(4000)], ignore_index = True)
df
df = df.sample(frac=1).reset_index(drop=True)
df
df['News'] = df['title'] + " " + df['text'] + " " + df['subject']
df
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
TL = WordNetLemmatizer()
corpus = []
for i in range (0,8000):
    line = re.sub('[^a-zA-Z]', " ", df['News'][i])
    line = line.lower()
    line = line.split()
    line = [TL.lemmatize(word) for word in line if not word in set(stopwords.words('english'))]
    line = " ".join(line)
    corpus.append(line)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = df['class'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
from sklearn.naive_bayes import GaussianNB 
model=GaussianNB()
final=model.fit(X_train,y_train)

print("Testing score",final.score(X_test,y_test))
import seaborn as sb
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

svm_predicted=final.predict(X_test)
svm_confuse=confusion_matrix(y_test,svm_predicted)
df_cm=pd.DataFrame(svm_confuse)

plt.figure(figsize=(5.5,4))
sb.heatmap(df_cm,annot=True,fmt='g')
plt.title("Confusion Matrix Heatmap")
plt.xlabel("True Label")
plt.ylabel("Predicted Label")
plt.show()
from sklearn.metrics import classification_report
print("Classification Report")
print(classification_report(y_test,svm_predicted))
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=2)
final=model.fit(X_train,y_train)

print("Testing score",final.score(X_test,y_test))

import seaborn as sb
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

knn_predicted=final.predict(X_test)
knn_confuse=confusion_matrix(y_test,knn_predicted)
df_cm=pd.DataFrame(knn_confuse)

plt.figure(figsize=(5.5,4))
sb.heatmap(df_cm,annot=True,fmt='g')
plt.title("Confusion Matrix Heatmap")
plt.xlabel("True Label")
plt.ylabel("Predicted Label")
plt.show()
from sklearn.metrics import classification_report
print("Classification Report")
print(classification_report(y_test,knn_predicted))