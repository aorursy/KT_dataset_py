import numpy as np 

import pandas as pd 

import os

from collections import defaultdict

import seaborn as sns

import matplotlib.pyplot as plt

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score,f1_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sub_df = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
train_df.head()
test_df.head()
train_df.isnull()
x=train_df.target.value_counts()

sns.barplot(x.index,x)

plt.gca().set_ylabel('samples')
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))

train_df_len1 = train_df[train_df['target']==1]['text'].str.len()

ax1.hist(train_df_len1,color='red')

ax1.set_title('disaster tweets')

train_Df_len2 = train_df[train_df['target']==0]['text'].str.len()

ax2.hist(train_Df_len2,color = 'green')

ax2.set_title('Non Disaster Tweets')
def create_corpus(target):

    corpus=[]

    

    for x in train_df[train_df['target']==target]['text'].str.split():

        for i in x:

            corpus.append(i)

    return corpus
corpus=create_corpus(0)



dic=defaultdict(int)

for word in corpus:

    if word in stop:

        dic[word]+=1

        

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 



for k in range(0,len(top)):

    print(top[k][0])
x,y = zip(*top)

print(x)

plt.bar(x,y)
x = train_df["text"]

y = train_df["target"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train
vect = CountVectorizer(stop_words = 'english')



x_train_cv = vect.fit_transform(X_train)

x_test_cv = vect.transform(X_test)

x_train_cv
clf = MultinomialNB()

clf.fit(x_train_cv, y_train)
pred = clf.predict(x_test_cv)
pred
y_test
# y_pred = model.predict(X_test)



f1score = f1_score(y_test,pred)

print(f"Model Score: {f1score * 100} %")
confusion_matrix(y_test, pred)
accuracy_score(y_test,pred)
y_test = test_df["text"]

y_test_cv = vect.transform(y_test)

preds = clf.predict(y_test_cv)
sub_df["target"] = preds

sub_df.to_csv("submission.csv",index=False)
sub_df.describe()
predicted = rf.predict(x_test_cv)

print(predicted)
y_test1 = test_df["text"]

y_test_cv = vect.transform(y_test1)

preds = rf.predict(y_test_cv)
sub_df["target"] = preds

sub_df.to_csv("submission_final.csv",index=False)