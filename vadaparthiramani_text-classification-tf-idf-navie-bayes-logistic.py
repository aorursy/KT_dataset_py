# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd

# data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) ; list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

#import neccessary Libraries

import nltk

import sklearn

import re

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

import string
#read train data and test data into a pandas Dataframe

train=pd.read_csv('../input/nlp-getting-started/train.csv')

test=pd.read_csv('../input/nlp-getting-started/test.csv')
#to differentiate train and test data maintain train lable as 1 for train data and 0 for test data

train['train']=1

test['train']=0
#concat train and test data for data cleaning

df=pd.concat([train,test],sort=True)
#convert the text into lower case letters to make classification irrespective of case

df['text']=df['text'].apply(lambda x: x.lower())
#Define a function clean_text remove links starts with https,htpps and remove line breaks,leading,trailing and extra spaces

def clean_text(text):

    text = re.sub(r'https?://\S+', '', text) # Remove link

    text = re.sub(r'http?://\S+', '', text) # Remove link

    text = re.sub(r'\n',' ', text) # Remove line breaks

    text = re.sub('\s+', ' ', text).strip() # Remove leading, trailing, and extra spaces

    return text
#apply clean_text function to the text column



df['text']=df['text'].apply(lambda x: clean_text(x))

#define a function to remove emojis in the text

def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)
#apply the remove_function to the text column

df['text']=df['text'].apply(lambda x: remove_emoji(x))
#define function which removes all special characters,numbers in the text expect alphabets 

values=list(string.ascii_lowercase)

def remover(my_string = ""):

  for item in my_string:

    if item not in values:

      my_string = my_string.replace(item, " ")

  return my_string
#apply the remover function to the text column

df['text']=df['text'].apply(lambda x: remover(x))
print(df['text'][df.id==40])

#replace the words having more than two letters with two letters.atleast the below code will convert the words like 'cooool' into 'cool'

#which converts it into correct spelling. 

for i in list(string.ascii_lowercase):

    df['text']=df['text'].apply(lambda x: re.sub('{}'.format(i)+'{3,}','{}'.format(i*2),x))

print(df['text'][df.id==40])

#Tried using spell checker and TextBlob for correcting the spellings of the words using below commented code but it is taking more and more time to run.
# !pip install pyspellchecker
# from spellchecker import SpellChecker



# spell = SpellChecker()

# def correct_spellings(text):

#     corrected_text = []

#     misspelled_words = spell.unknown(text.split())

#     for word in text.split():

#         if word in misspelled_words:

#             corrected_text.append(spell.correction(word))

#         else:

#             corrected_text.append(word)

#     return " ".join(corrected_text)

        

# text = "corect me plese"

# correct_spellings(text)
# df['text']=df['text'].apply(lambda x: correct_spellings(x))
#replaces the single letter words which may not be useful for classification

df['text']=df['text'].apply(lambda x: re.sub(' [a-z] ',' ',x))
#replace the words not having vowels 'aeiou' and semi vowel 'y' with empty string as there are no words in english which dont have vowels or semi vowel.

#the below code when I apply to text cloumn it removes the words like 'bbcmtd' which are not in english dictionary.

print(df['text'][df.id==48])

df['text']=df['text'].apply(lambda x: re.sub('\\b[bcdfghjklmnpqrstvwxz]+\\b',' ',x))

print(df['text'][df.id==48])
df['text']=df['text'].apply(lambda x: re.sub('\s+', ' ', x).strip()) # Remove leading, trailing, and extra spaces)
#define the function which tokenizes the text and removes the stop words like 'is,a,the' in the english which were common in every text and are not useful for classification.

#Lemmatize the each word which converts the -ing forms or -ed forms into verb V1 form

words=nltk.corpus.stopwords.words("english")

def custom_tokenizer(str):

    lemmatizer = WordNetLemmatizer()

    tokens = nltk.word_tokenize(str)

    remove_stopwords = list(filter(lambda token: token not in words,tokens))

    lematize_words = [lemmatizer.lemmatize(word) for word in remove_stopwords]

    return lematize_words
#Tokenize the data using TFIDF vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)

tfidf = vectorizer.fit_transform(df['text'])

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

#split into training and test dataset

x_train,x_test,y_train,y_test = train_test_split(tfidf[:train.shape[0]],df['target'][df.train==1],test_size = 0.2,random_state=0)

#Model Building using Multinomial Navie Bayes

classifiernb = MultinomialNB()

#Fit the nodel with the training and testing instances from train data

classifiernb.fit(x_train,y_train)
#Predict the outcome and print its evaluation metrics 

pred = classifiernb.predict(x_test)

print('test dataset confusion matrix:',sklearn.metrics.confusion_matrix(y_test,pred),"\n")

print('test dataset accuracy score:',sklearn.metrics.accuracy_score(y_test,pred),"\n")

print('test dataset classification report:\n',sklearn.metrics.classification_report(y_test,pred),"\n")

print('test dataset f1_score',sklearn.metrics.f1_score(y_test,pred),"\n")

print('test dataset r2_score',sklearn.metrics.r2_score(y_test,pred),"\n")
from sklearn.linear_model import LogisticRegression
#Model Building using Logistic Regression

classifier_lr = LogisticRegression()

classifier_lr.fit(x_train, y_train)
#Predict the outcome and print its evaluation metrics 

pred_lr = classifier_lr.predict(x_test)

print('test dataset confusion matrix:',sklearn.metrics.confusion_matrix(y_test,pred_lr),"\n")

print('test dataset accuracy score:',sklearn.metrics.accuracy_score(y_test,pred_lr),"\n")

print('test dataset classification report:\n',sklearn.metrics.classification_report(y_test,pred_lr),"\n")

print('test dataset f1_score',sklearn.metrics.f1_score(y_test,pred_lr),"\n")

print('test dataset r2_score',sklearn.metrics.r2_score(y_test,pred_lr),"\n")
classifiernaive = MultinomialNB()

#Fit the Naive Bayes classifier with train dataset

classifiernaive.fit(tfidf[:train.shape[0]],df['target'][df.train==1])
#predict the unknown outcomes for target whether it is a Disaster or not

pred_test = np.round(classifiernaive.predict(tfidf[train.shape[0]:])).astype('int')
pred_target=pd.DataFrame(pred_test,columns=['target']).reset_index().rename(columns={'index':'index_id'})
pred_id=pd.DataFrame(df['id'].iloc[train.shape[0]:]).reset_index().rename(columns={'index':'index_id'})
#store the predictions with id in submission dataframe

submission=pd.merge(pred_id,pred_target,left_on='index_id',right_on='index_id').drop(columns=['index_id'])
submission.head()
submission.to_csv('submission.csv',index=False)