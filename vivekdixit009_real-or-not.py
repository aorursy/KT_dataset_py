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
import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')

#importing datasets

tweet =pd.read_csv(r'/kaggle/input/nlp-getting-started/train.csv',encoding='utf-8')


X,y = tweet.text, tweet.target
corpus =[]

for i in range (0,len(X)):
    rep= re.sub(r'\W',' ',str(X[i])) #replacing the non words with spaces
    rep = rep.lower()
    rep = re.sub(r's+[a-z]\s+',' ',rep)  #removing all the single character
    rep = re.sub(r'^[a-z]\s+',' ',rep) #removing the single charater from start
    rep = re.sub(r'\s+',' ',rep) #removing the extra generated extra spaces
    corpus.append(rep)
    from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=3000,min_df=3,max_df=0.6,stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()


#calculating TFIDF score
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()
#splitting the dataset in traing and testing

from sklearn.model_selection import train_test_split
#test size is 20%
text_train,text_test,sent_train,sent_test = train_test_split(X,y,test_size=.2,random_state=0)

#setting up logistic regression

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(text_train,sent_train)

#making prediction

sent_pred = classifier.predict(text_test)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(sent_pred,sent_test)
cm
test_tweet = pd.read_csv(r'/kaggle/input/nlp-getting-started/test.csv',encoding='utf-8')

t1=test_tweet.text

corpus1 =[]

for i in range (0,len(t1)):
    rep= re.sub(r'\W',' ',str(X[i])) #replacing the non words with spaces
    rep = rep.lower()
    rep = re.sub(r's+[a-z]\s+',' ',rep)  #removing all the single character
    rep = re.sub(r'^[a-z]\s+',' ',rep) #removing the single charater from start
    rep = re.sub(r'\s+',' ',rep) #removing the extra generated extra spaces
    corpus1.append(rep)
    
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=3000,min_df=3,max_df=0.6,stop_words=stopwords.words('english'))
t1 = vectorizer.fit_transform(corpus).toarray()


#calculating TFIDF score
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
t1 = transformer.fit_transform(X).toarray()


test_csv_pred = classifier.predict(t1)
test_csv_pred
# for i in range(len(test_csv_pred)):
#     test_tweet['id'][i]=test_csv_pred[i]
    
p=test_tweet['id']
#test_tweet.to_csv('/kaggle/output/nlp-getting-started/final_upload.csv', index=False) 
df['id']=p
for i in range(len(test_csv_pred)):
     df['target'][i]=test_csv_pred[i]
df.to_csv('/kaggle/working/final_upload.csv', index=False) 
