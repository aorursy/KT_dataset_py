# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dfS08_question= pd.read_csv('../input/S08_question_answer_pairs.txt',  sep="\t")
dfS08_question= pd.read_csv('../input/S08_question_answer_pairs.txt',  sep="\t")

dfS08_question.drop(labels=['ArticleTitle', 'DifficultyFromQuestioner', 'DifficultyFromAnswerer', 'ArticleFile', 'Answer'], axis=1, inplace=True)

dfS08_question.head()
import nltk

import numpy as np

from nltk.tokenize import sent_tokenize



dfS08_question.to_csv("./csvfile.txt", sep=";", index=False, header=False)

print(os.listdir("."))



rawText = np.genfromtxt("./csvfile.txt", dtype='str', delimiter=';', usecols=np.arange(0,1))

csvFile.shape

from nltk import pos_tag

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

import string

from nltk.corpus import wordnet



stopwords_list = stopwords.words('english')



lemmatizer = WordNetLemmatizer()



def my_tokenizer(doc):

    words = word_tokenize(doc)

    

    pos_tags = pos_tag(words)

    

    non_stopwords = [w for w in pos_tags if not w[0].lower() in stopwords_list]

    

    non_punctuation = [w for w in non_stopwords if not w[0] in string.punctuation]

    

    lemmas = []

    for w in non_punctuation:

        if w[1].startswith('J'):

            pos = wordnet.ADJ

        elif w[1].startswith('V'):

            pos = wordnet.VERB

        elif w[1].startswith('N'):

            pos = wordnet.NOUN

        elif w[1].startswith('R'):

            pos = wordnet.ADV

        else:

            pos = wordnet.NOUN

        

        lemmas.append(lemmatizer.lemmatize(w[0], pos))



    return lemmas

    

    
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf_vectorizer = TfidfVectorizer(tokenizer=my_tokenizer)



tfs = tfidf_vectorizer.fit_transform(rawText)



print(tfs.shape)
from sklearn.metrics.pairwise import cosine_similarity

while True:

    answer = input("Digite sua pesquisa")

    if answer == 'Sair':

        break

    elif answer !='Sair':

        query_vect = tfidf_vectorizer.transform([answer])

        positions = cosine_similarity(query_vect, tfs)[0]

        print('O texto mais proximo localizado foi: ', rawText[np.argmax(positions)])

        

