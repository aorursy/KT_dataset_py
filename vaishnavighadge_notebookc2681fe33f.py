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
df=pd.read_csv('/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv')
df.head()
import re

import nltk 

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer

ps=PorterStemmer()

wordnet=WordNetLemmatizer()

collect = []

for i in range(0, len(df)):

    values = re.sub('[^a-zA-Z]', ' ', df['Message'][i])

    values = values.lower()

    values = values.split()

    values = [wordnet.lemmatize(word) for word in values if not word in stopwords.words('english')]

    values = ' '.join(values)

    collect.append(values)

    
df['Category'].replace({'ham':1,'spam':0},inplace=True)
df['Category']
y=df['Category']

y
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
from sklearn.svm import SVC

model=SVC()

model
model.fit(X_train,y_train)
from sklearn.feature_extraction.text import CountVectorizer

cs=CountVectorizer(max_features=1000)

X=cs.fit_transform(collect).toarray()
X
y_pred=model.predict(X_test)

y_pred
from sklearn.metrics import accuracy_score,confusion_matrix

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))