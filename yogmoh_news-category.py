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
!pip install -U texthero
import tensorflow as tf

import pandas as pd

import numpy as np

import texthero as hero



import matplotlib.pyplot as plt

import re

import matplotlib as mpl



import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

mpl.rcParams['figure.dpi'] = 300
train = pd.read_excel('/kaggle/input/new-category/Data_Train.xlsx')

test= pd.read_excel('/kaggle/input/new-category/Data_Test.xlsx')



display(train.sample(5))

display(train.info())

display(test.info())
train.STORY[0]
train.SECTION.value_counts(normalize=True)
combined_df = pd.concat([train.drop('SECTION',axis=1),test])

combined_df.info()
hero.visualization.wordcloud(combined_df['STORY'], max_words=1000,background_color='BLACK')
hero.Word2Vec
combined_df['cleaned_text']=(combined_df['STORY'].pipe(hero.remove_angle_brackets)

                    .pipe(hero.remove_brackets)

                    .pipe(hero.remove_curly_brackets)

                    .pipe(hero.remove_diacritics)

                    .pipe(hero.remove_digits)

                    .pipe(hero.remove_html_tags)

                    .pipe(hero.remove_punctuation)

                    .pipe(hero.remove_round_brackets)

                    .pipe(hero.remove_square_brackets)

                    .pipe(hero.remove_stopwords)

                    .pipe(hero.remove_urls)

                    .pipe(hero.remove_whitespace)

                    .pipe(hero.lowercase))
lemm = WordNetLemmatizer()



def word_lemma(text):

    words = nltk.word_tokenize(text)

    lemma = [lemm.lemmatize(word) for word in words]

    joined_text = " ".join(lemma)

    return joined_text
combined_df['lemmatized_text'] = combined_df.cleaned_text.apply(lambda x: word_lemma(x))
text = []

for i in range(len(combined_df)):

    review = nltk.word_tokenize(combined_df['lemmatized_text'].iloc[i])

    review = ' '.join(review)

    text.append(review)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,f1_score,plot_confusion_matrix



from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
X = combined_df['lemmatized_text'].iloc[:7628]

test_df = combined_df['lemmatized_text'].iloc[7628:]
cv = CountVectorizer(max_features=9000)

cv.fit(X)

X = cv.transform(X)

test_df = cv.transform(test_df)



y = train.SECTION
tfid = TfidfVectorizer(max_features=9000)

tfid.fit(X)

X = tfid.transform(X)

test_df = tfid.transform(test_df)



y = train.SECTION
from imblearn.over_sampling import SMOTE

smote = SMOTE()

X_new,y_new = smote.fit_resample(X,y)
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42,stratify=y_new)
from sklearn.ensemble import RandomForestClassifier



rf=RandomForestClassifier()

rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

accuracy_score(y_test,y_pred)
from xgboost import XGBClassifier



xg=XGBClassifier()

xg.fit(X_train,y_train)

y_pred = xg.predict(X_test)

accuracy_score(y_test,y_pred)
from catboost import CatBoostClassifier

cat=CatBoostClassifier(task_type='GPU')

cat.fit(X_train,y_train)

y_pred = cat.predict(X_test)

accuracy_score(y_test,y_pred)
predictions=xg.predict(test_df)

submissions = pd.DataFrame({'SECTION':predictions})

submissions.to_csv('./sub8.csv',index=False,header=True)