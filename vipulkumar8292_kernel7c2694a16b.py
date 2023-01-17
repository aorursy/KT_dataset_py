import numpy as np
import pandas as pd
msg = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding='ISO-8859-1')
msg.head()
msg.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
msg.rename(columns={'v1':'label','v2':'message'},inplace=True)
msg.head()
y = msg['label']
x = msg['message']
import re
x_new = []
for w in x:
    w = re.sub('[^a-zA-Z]',' ',str(w))
    w.lower()
    x_new.append(w)
x_new
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_new,y,test_size=0.25,random_state=22)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=6).fit(x_train)
len(tfidf.get_feature_names())
from sklearn.linear_model import LogisticRegression
logis = LogisticRegression()
x_train_vector = tfidf.transform(x_train)
logis.fit(x_train_vector,y_train)
prediction_logis = logis.predict(tfidf.transform(x_test)) 
from sklearn.metrics import confusion_matrix,classification_report #importing confusion matrix
print(confusion_matrix(prediction_logis,y_test))
print(classification_report(prediction_logis,y_test)) #error report
m1 = 'Or ill be a little closer like at the bus stop on the same street'
print(logis.predict(tfidf.transform([m1])))
