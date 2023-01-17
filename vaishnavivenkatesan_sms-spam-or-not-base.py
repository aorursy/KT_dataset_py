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
FILEPATH = '/kaggle/input/sms-spam-collection-dataset/spam.csv'
df = pd.read_csv(FILEPATH, encoding='iso-8859-1', engine = 'c') # engine 'c' used instead of 'python' for higher performance
df.head(10)
# delete unnecessary cols
cols = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']

df.drop(cols, axis = 1, inplace = True)
df.head()
# Title change v1 = result, v2 = input

df.rename(columns={'v1':'Classification','v2':'Sms content'},inplace=True)

# we can also use df.rename() option here
df.head()
# reorder options - must be applicable for all cols
df = df[['Sms content','Classification']]
 
df.head()
df.head()
df.count()

df.describe()
# print first string

df.iloc[0][0]
df.iloc[2][0]
def message_length(msg):
    
    msg_words = msg.split(' ')
    
    msg_len = len(msg_words)
    
    return msg_len
print(message_length(df.iloc[1][0]))
# Create a new col called 'message_word_length' showing how many words in the message
df['words_count'] = df['Sms content'].apply(message_length)
df.head()


# ref: https://rajacsp.github.io/mlnotes/python/data-wrangling/advanced-custom-lambda/
# show the unique labels

df['Classification'].unique()
print(find_length(df.iloc[0][0]))
# Create a new col called 'message_word_length' showing how many words in the message
df['msg_length'] = df['Sms content'].apply(lambda x:len(x))
df.head()
# History words count

import matplotlib.pyplot as plt

# to avoid popups use inline
%matplotlib inline 
# plt.hist(data['label'], bins=3, weights=np.ones(len(data['label'])) / len(data['label']))

import numpy as np

plt.hist(df['words_count'], bins = 50, alpha=0.5)
plt.hist(df['msg_length'], bins=50 ,alpha=0.3)
plt.xlabel('Word Length')
plt.ylabel('Group Count')
plt.title('Word Length Histogram')
# Find more than 80 words
df['words_count']
df1 = df[(df['words_count'] > 80) & (df['msg_length'] > 100)]

df1
# encode
from sklearn import preprocessing

encode = preprocessing.LabelEncoder()
df['find spam'] = encode.fit_transform(df['Classification'])
df
import string
string.punctuation
def remove_punctuation(text):
    new_text=''.join([char for char in text if char not in string.punctuation])
    return new_text
df['new_sms']=df['Sms content'].apply(lambda row : remove_punctuation(row))
df
import re
def tokenize(text):
    tokens=re.split('\W+',text)
    return tokens 
df['tokenized_text']=df['new_sms'].apply(lambda row : tokenize(row.lower()))
df.head()
import nltk
stopwords=nltk.corpus.stopwords.words('english')
stopwords[:5]
def remove_stopwords(text):
    clean_text=[word for word in text if word not in stopwords]
    return clean_text 
df['clean_text'] = df['tokenized_text'].apply(lambda row : remove_stopwords(row))
df.head()
ps = nltk.PorterStemmer()
dir(ps)

from nltk.stem import PorterStemmer
def stemming(tokenized_text):
    stemmed_text=[ps.stem(word) for word in tokenized_text]
    return stemmed_text
df['stemmed_text']=df['clean_text'].apply(lambda row : stemming(row))
df[['Sms content','stemmed_text']].head()
def get_final_text(stemmed_text):
    final_text=" ".join([word for word in stemmed_text])
    return final_text
df['final_text']=df['stemmed_text'].apply(lambda row : get_final_text(row))
df.head()