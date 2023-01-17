

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB

import emoji, re



train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv').set_index('id')

test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv').set_index('id')
#Combine Test and Train

df=train.append(test)
# Work for both emojis and special content.



def emoji_extraction(sent):

    e_sent = emoji.demojize(sent)

    

    return re.findall(':(.*?):',e_sent)

def emoji_count(sent):

    e_sent = emoji.demojize(sent)

    return len(re.findall(':(.*?):',e_sent))



def emoji_to_text(sent):

    e_sent = emoji.demojize(sent)

    emo = re.findall(':(.*?):',e_sent)

    for e in emo:

        e_sent = e_sent.replace(':{}:'.format(e),'{}'.format(e))

    return e_sent
%%time

df['text'] = df['text'].apply(emoji_to_text)
#Remove URLs

def urls(sent):

    return re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',sent)

def url_counts(sent):

    return len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',sent))

def remove_urls(sent):

    return re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',sent)
%%time

df['text'] = df['text'].apply(remove_urls)
#Remove stopwords and Punctuations 

import spacy

nlp = spacy.load('en',disable=['parser', 'tagger','ner'])



all_stopwords = (nlp.Defaults.stop_words)





sw_and_punc=' '.join(all_stopwords)+ ' '+'\n\n \n\n\n! " - # $ % & ( ) -- . * + , -/ : ; < = > ? @ [ \\ ] ^ _ ` { | } ~ \t \n  ... / http//t.co '



sw_and_punc=sw_and_punc.split()



sw_and_punc.append("'")
def removeSW(inString):

    

    text_list=[token.text.lower() for token in nlp(inString) if token.text.lower() not in sw_and_punc]    

    return(' '.join(text_list))



import re

def findHashTag(txt):

    

    hash_tags =re.findall(r"#(\w+)", txt)

    return hash_tags
df['text']=df['text'].apply(removeSW)


df['temp_list'] = df['text'].apply(lambda x:str(x).split())

top = Counter([item for sublist in df['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(20))

temp.columns = ['Common_words','count']

temp.style.background_gradient(cmap='Blues')
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm=tfidf.fit_transform(df['text'])

test_dtm=dtm[7613:]

train_dtm=dtm[:7613]

X=train_dtm

y=train['target']

from sklearn.model_selection import GridSearchCV

svc = SVC()

parameters = {'kernel':('linear', 'rbf'), 'C':[0.8,0.9,1,1.1,1.2,1.4]}

clf = GridSearchCV(svc, parameters)

clf.fit(X, y)

clf.best_params_
model=SVC(C=1, kernel='rbf')



model.fit(X,y)



prediction=model.predict(test_dtm)

test['target']=prediction




model_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

model_submission['target'] = np.round(prediction).astype('int')

model_submission.to_csv('model_submission.csv', index=False)

model_submission.describe()