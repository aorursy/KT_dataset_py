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
import seaborn as sns
train_df = pd.read_csv('../input/detect-the-sentiments/train_2kmZucJ.csv')
test_df = pd.read_csv('../input/detect-the-sentiments/test_oJQbWVk.csv')
train_df.head()
test_df.head()
train_df['tweet'][0]
train_df['tweet'][7919]
text = train_df['tweet']
text
test_text = test_df['tweet']
test_text
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

wordnet = WordNetLemmatizer ()
corpus = []
for i in range(0,len(text)):
    review = re.sub('[^a-zA-Z]',' ',text[i])  #re.sub(pattern, repl, string) - substituting characters apart from a-z,A-Z with blank space
    review = review.lower()
    review = review.split()#splitting the words
    
    review = [wordnet.lemmatize(word)  for word in review if not word in stopwords.words('english')] #for words not in stopwrods we are obtaining the root word of the words throguh stemmer(eg. preference - prefer)
    review = ' '.join(review)
    corpus.append(review)
    
    
test_corpus = []
for i in range(0,len(test_text)):
    review = re.sub('[^a-zA-Z]',' ',text[i])  #re.sub(pattern, repl, string) - substituting characters apart from a-z,A-Z with blank space
    review = review.lower()
    review = review.split()#splitting the words
    
    review = [wordnet.lemmatize(word)  for word in review if not word in stopwords.words('english')] #for words not in stopwrods we are obtaining the root word of the words throguh stemmer(eg. preference - prefer)
    review = ' '.join(review)
    test_corpus.append(review)

corpus[2]
test_corpus[10]
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v = TfidfVectorizer(max_features=5000,ngram_range=(1,3))
X = tfidf_v.fit_transform(corpus).toarray()
y = train_df['label']
test = tfidf_v.fit_transform(test_corpus).toarray()
X.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
tfidf_v.get_feature_names()[:20] #this is for the training text #the top 20 words
tfidf_v.get_params() #for the training text
count_df = pd.DataFrame(X_train,columns=tfidf_v.get_feature_names())
count_df
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,f1_score
classifier = MultinomialNB()
classifier.fit(X_train,y_train)

pred = classifier.predict(X_test)
sns.heatmap(confusion_matrix(y_test,pred),annot = True)
f1_score(y_test,pred)
from sklearn.linear_model import PassiveAggressiveClassifier
lnr_classifier = PassiveAggressiveClassifier(n_iter_no_change=50)
lnr_classifier.fit(X_train,y_train)
pred2 = lnr_classifier.predict(X_test)
sns.heatmap(confusion_matrix(y_test,pred2),annot=True)
f1_score(y_test,pred2)
pred_final = classifier.predict(test)
submission = pd.DataFrame()
submission['id'] = test_df['id']
submission['label'] = pred_final
submission
submission['label'].value_counts()
