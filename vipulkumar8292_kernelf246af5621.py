import numpy as np

import pandas as pd
store = pd.read_csv('../input/google-play-store-apps/googleplaystore_user_reviews.csv')
store.head()
store.shape
store.isnull().sum()
store.dropna(inplace=True)
store.shape
from sklearn.preprocessing import LabelEncoder

label_enc = LabelEncoder()
df = (np.array(store['Sentiment'])).reshape(-1,1)           
df.shape
from sklearn.preprocessing import LabelEncoder

label_enc = LabelEncoder()
df = pd.DataFrame(df)
store['Sentiment'] = df.apply(label_enc.fit_transform)
store['Sentiment'].dropna(inplace=True)
store['Sentiment'].isnull().sum()
store['Sentiment'].astype(int)
store.dropna(inplace=True)
store.isnull().sum()
x = store['Translated_Review']

y = store['Sentiment']
import re

x_new = []

for w in x:

    w = re.sub('[^a-zA-Z ]','',str(w))

    w = w.lower()

    x_new.append(w)
x_new
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_new,y,test_size=0.25,random_state=35)
from sklearn.feature_extraction.text import TfidfVectorizer #importing tf-idf
bags = TfidfVectorizer(min_df=6).fit(x_train) #minimum document frequency=6, rare word is neglected
x_train_vector = bags.transform(x_train)
x_train_vector.toarray().shape
len(bags.get_feature_names())
from sklearn.linear_model import LogisticRegression
logis = LogisticRegression()
x_train_vector = bags.transform(x_train) #converting into tf-idf vector
logis.fit(x_train_vector,y_train)
prediction_logis = logis.predict(bags.transform(x_test)) 
from sklearn.metrics import confusion_matrix,classification_report #importing confusion matrix
print(confusion_matrix(prediction_logis,y_test))
print(classification_report(prediction_logis,y_test)) #error report