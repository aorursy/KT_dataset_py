# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test_data = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

test_data.head()
train_data = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

train_data.head()
import spacy

from spacy.lang.en.stop_words import STOP_WORDS

import re

import string

from sklearn.naive_bayes import MultinomialNB



# Processing the data

def Text_process(text):

    nlp = spacy.load('en')

    

    temp=[]

    for i in text:

        text_each=[]

        i=i.lower()

        i=re.sub(r'\d+', '', i)

        for each in nlp(i):

            if len(each) > 2 and each.is_stop==False and 'http' not in each.text and '@' not in each.text and '.' not in each.text and '' not in each.text and ' ' not in each.text and '-' not in each.text and '(' not in each.text and ')' not in each.text and '/' not in each.text and '&' not in each.text and "'" not in each.text:

                text_each.append(each.text)

        temp.append(text_each)

    return temp





# Get the set of words

def Word_set(text):

    word_set=set()



    for line in text:

        for word in line:

            word_set.add(word)

    return word_set





# Word frequency statistics

def Word_dict(text):

    all_words_dict = {}

    for word_list in text:

        for word in word_list:

            if word in all_words_dict.keys():

                all_words_dict[word] += 1

            else:

                all_words_dict[word] = 1

                

    all_words_tuple_list = sorted(all_words_dict.items(), key = lambda f:f[1], reverse = True)

    all_words_list, all_words_nums = zip(*all_words_tuple_list)

    all_words_list = list(all_words_list)

    

    return all_words_dict



# Text vectorization

def TextFeatures(data,feature_words):

    features=[]

    for i in data:

        temp=[]

        for word in feature_words:

            if word in i:

                temp.append(1)

            else:

                temp.append(0)

        features.append(temp)

    return features
model=MultinomialNB()



train_text=Text_process(train_data['text'])

train_label=train_data['target']



test_text=Text_process(test_data['text'])



feature_words=Word_set(train_text)
train_feature=TextFeatures(train_text, feature_words)



test_feature=TextFeatures(test_text, feature_words)
classifier = model.fit(train_feature, train_label)
test_label = model.predict(test_feature)



output = pd.DataFrame({'id': test_data['id'], 'target': test_label})



print(output.head(10))
output.to_csv('my_submission.csv', index=False)



print("Your submission was successfully saved!")