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

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd

from collections import Counter

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfTransformer
df_train=pd.read_csv("/kaggle/input/janatahack-independence-day-2020-ml-hackathon/train.csv")

df_test=pd.read_csv("/kaggle/input/janatahack-independence-day-2020-ml-hackathon/test.csv")



df_train.head()
df_train["text"]=df_train["TITLE"]+df_train["ABSTRACT"]

df_test["text"]=df_test["TITLE"]+df_test["ABSTRACT"]

del df_train["TITLE"]

del df_train["ABSTRACT"]

del df_train["ID"]

main_test_ids=df_test["ID"]

main_test_title=df_test["TITLE"]

main_test_abstract=df_test["ABSTRACT"]

del df_test["TITLE"]

del df_test["ABSTRACT"]

del df_test["ID"]



df_train.head()
df_train["text"][1]

df_train_classes=df_train.drop("text",axis=1)

df_train_classes.head()
import re

def preprocess_text(sen):

    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    sentence = re.sub(r'\s+', ' ', sentence)



    return sentence
X = []

sentences = list(df_train["text"])

for sen in sentences:

    X.append(preprocess_text(sen))

df_train["text"]=X

X=[]

sentences = list(df_test["text"])

for sen in sentences:

    X.append(preprocess_text(sen))

df_test["text"]=X
X_train, X_test, y_train, y_test = train_test_split(df_train["text"],df_train_classes, test_size=0.2, random_state=42)
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC,SVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



labels = ['Computer Science', 'Physics', 'Mathematics','Statistics','Quantitative Biology', 'Quantitative Finance']

label_predict=[]

for label in labels:

    text_clf = Pipeline([('tfidf', TfidfVectorizer(min_df=True,smooth_idf=True,sublinear_tf=True,analyzer='char',strip_accents='ascii', ngram_range=(4,8))),

                         ('clf',LinearSVC(loss="hinge",fit_intercept=False ,class_weight='balanced')),

    ])

# LinearSVC(loss="hinge",fit_intercept=False,tol=1e-3)

    text_clf.fit(X_train, y_train[label])  

# min_df=True,smooth_idf=True,sublinear_tf=True,analyzer='char',strip_accents='ascii',token_pattern=r'(?ui)\\b\\w*[a-z]+\\w*\\b', ngram_range=(4,8)

    predictions = text_clf.predict(X_test)



#     naive_bayes = MultinomialNB()

#     naive_bayes.fit(X_train_cv, y_train[label])

#     predictions = naive_bayes.predict(X_test_cv)

    

    print("Accuracy score: ", accuracy_score(y_test[label], predictions))

    print("Precision score: ", precision_score(y_test[label], predictions))

    print("Recall score: ", recall_score(y_test[label], predictions))

    

    final_predict= text_clf.predict(df_test["text"])

    print(final_predict)

    label_predict.append(np.array(final_predict))



label_predict.append(np.array(main_test_ids))

for x in label_predict:

    print(len(x))
dataset = pd.DataFrame({'ID':label_predict[6],'Computer Science': label_predict[0],'Physics': label_predict[1],'Mathematics': label_predict[2],'Statistics': label_predict[3],'Quantitative Biology': label_predict[4],'Quantitative Finance': label_predict[5]})
dataset.head()
dataset.to_csv(r'submission.csv', index = False)