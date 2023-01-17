import pandas as pd

import string

import numpy as np

from nltk.corpus import stopwords

from nltk import sent_tokenize, word_tokenize, pos_tag

from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer

from nltk.probability import FreqDist



import gensim

from gensim.parsing.preprocessing import STOPWORDS



import sys

import os

print(os.listdir("../input/nlp"))
train = pd.read_csv("../input/nlp/train.csv")

print("Original Shape => {} ".format(train.shape))

train['question1'].replace('', np.nan, inplace=True)

train['question2'].replace('', np.nan, inplace=True)

train.dropna(subset=['question1', 'question2'], inplace=True)

print("New Shape => {} ".format(train.shape))

train.head()
def tokenize_word(questions):

    words = []

    for eachQues in questions:

        #print(eachQues)

        words.append(word_tokenize(eachQues))

    return words



question1 = train['question1']

word1 = tokenize_word(question1)



question2 = train['question2']

word2 = tokenize_word(question2)
def stem_lemmat(token):

    return PorterStemmer().stem(WordNetLemmatizer().lemmatize(token))



stopWords = set(stopwords.words('english') + list(STOPWORDS) + list(string.punctuation))



def process(question):

    processedQues = []

    processedWord = []

    for eachQues in question:

        no_stpwrds = [word.lower() for word in eachQues if word.lower() not in stopWords]

        stem_lemmatise = [stem_lemmat(word) for word in no_stpwrds]

        processedWord.append(stem_lemmatise)

        processedQues.append(" ".join(stem_lemmatise))

    return processedWord, processedQues



processed_word1, processed_question1 = process(word1)

processed_word2, processed_question2 = process(word2)

duplicate_copy = train['is_duplicate']

id_copy = train['id']

qid1_copy = train['qid1']

qid2_copy = train['qid2']

dataset = pd.DataFrame({'id':id_copy,'qid1':qid1_copy, 'qid2':qid2_copy,'question1': processed_question1,\

                        'question2':processed_question2, 'is_duplicate': duplicate_copy}, \

                       columns=['id','qid1', 'qid2', 'question1', 'question2', 'is_duplicate'])

dataset.head()