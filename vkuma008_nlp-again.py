corpus = [
    "This is my first corpus",
    "Processing it for ML",
    "Doing ML is awesome",
    "This is fun to look at",
    "ML is life, ML is interest"
]
corpus

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus)
X
X.toarray()
cv.get_feature_names()
cv.vocabulary_
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.DataFrame({'Text':['This is my first text! on today', 'I am working oon text data:)', 'I am creating, text data frame'],'Label':[1,1,0]})
df
df['Text'].str.get_dummies(' ')
a = cv.fit_transform(df['Text']).toarray()
b = cv.get_feature_names()
pd.DataFrame(a,columns=b)
df['Text'] = df['Text'].str.lower().str.replace('[^a-z]',' ').str.split()
df
import nltk
from nltk.corpus import stopwords
df['Text'] = df['Text'].apply(lambda x: [word for word in x if word not in set(stopwords.words('english'))])
df
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps
df['Text'] = df['Text'].apply(lambda x: ' '.join([ps.stem(word) for word in x]))
df
pd.DataFrame(cv.fit_transform(df['Text']).toarray(),columns=cv.get_feature_names())
spam_df = pd.read_csv('spam.csv',encoding='ISO-8859-1',engine='c')
spam_df.head()
spam_df = spam_df.loc[:,['v1','v2']]
spam_df.head()
spam_df.rename(columns={'v1':'Target','v2':'Text'},inplace=True)
spam_df.head()
from wordcloud import WordCloud
spam_list = spam_df[spam_df['Target']=='spam']['Text'].unique().tolist()
spam_list[:2]
spam = ' '.join(spam_list)
spam[:100]
spam_wc = WordCloud().generate(spam)
spam_wc
plt.figure()
plt.imshow(spam_wc)
plt.show()
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


ps = PorterStemmer()

spam_df['Text'] = spam_df['Text'].str.lower().str.replace('[^a-z]', ' ').str.split()
spam_df['Text']
spam_df['Text'] = spam_df['Text'].apply(lambda x: ' '.join([ps.stem(word) for word in x if word not in set(stopwords.words('english'))]))
spam_df.head()
X = cv.fit_transform(spam_df.Text).toarray()
X.shape
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y= le.fit_transform(spam_df.Target)
y[:5]
y.shape
le.classes_
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .25, random_state = 0)
clf = MultinomialNB()
clf
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
test_data = spam_df.head(10).Text.tolist()+spam_df.tail(10).Text.tolist()
pred_data = spam_df.head(10).Target.tolist()+spam_df.tail(10).Target.tolist()
test_pred = clf.predict(cv.transform(test_data))
test_pred
i = 0
for sms, label in zip(test_data,pred_data):
    print(str(test_data[i][:50])+"("+str(pred_data[i])+")=>"+str(test_pred[i]))
    i+=1
