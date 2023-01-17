# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import json

from pprint import pprint

import nltk

from nltk import word_tokenize

from nltk.tokenize import sent_tokenize

from nltk.util import ngrams

import re

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_json=[]

with open(os.path.join(dirname, filename),'r',encoding='utf8') as file:

        for line in file:

            data_json.append(json.loads(line))
title = []

article = []

for item in data_json:

   # pprint(item['title'])

    title.append(item['title'])

    article.append(item['data'])

pprint(len(title))

pprint(len(article))

pprint(title[1])

title_tokenized = []

for item in title:

    s=item

    s = s.lower()

    s = re.sub(r"\b[\u0600—\u06FF]+#(?=\s|$)", ' ', s)

    tokens = [token for token in s.split(" ") if token != ""]

    title_tokenized.append(tokens)

pprint(len(title_tokenized))            
pprint(title_tokenized)
stopwords = nltk.corpus.stopwords.words('english')

puncList = [".","-",";",":","!","?","/","\\",",","#","@","$","&",")","(","\"","``","'s","''","'","d.","'the","...","’","“","”","so..","–"]

dayMonthList = ["mon","tue","wed","thurs","fri","sat","sun","monday","tuesday","wednesday","thursday","friday","saturday","sunday","jan","januaury","feb","february","mar","march","apr","april","may","may","jun","june","jul","july","aug","august","sept","september","oct","october","nov","november","dec","december"]

wordList=["benefit:","audio:","article","seminar:","workshop","part","1","2","3","4","5","6","7","8","9","0"]

title_cleaned=[]

for item in title_tokenized:

    title_cleaned.append([x for x in item if x.lower() not in stopwords and x not in puncList and x.lower() not in dayMonthList and x.lower() not in wordList])

pprint(len(title_cleaned)) 
pprint(title_cleaned)
count=0

total=1814

article_index =[]

for art in article:

    if re.search(r'Download Ad',art) or re.search(r'Audio',art) or re.search(r'Seminar',art) or re.search(r'Workshop',art):

        count=count +1

        index=article.index(art)

        article.pop(index)

        title_cleaned.pop(index)

        article_index.append(index)

total = total-count

print(total)

print(article_index)

print(1814 - len(article_index))
'''for item in title_cleaned:

    if title_cleaned.index(item) in article_index:

        del title_cleaned[title_cleaned.index(item)]

for item in article:

    if article.index(item) in article_index:

        del article[article.index(item)]

'''

pprint(len(title_cleaned))

pprint(len(article))
title_bigrams=[]

title_trigrams=[]

for item in title_cleaned:

    tokens = [word for word in item]

    title_bigrams.append(list(ngrams(tokens, 2)))

    title_trigrams.append(list(ngrams(tokens,3)))

pprint(len(title_bigrams))

pprint(len(title_trigrams))
for item in title_bigrams:

    pprint(item)
for item in title_trigrams:

    pprint(item)
pprint(title_bigrams[0])

pprint(article[0])
data_bigrams={}

for i in range(len(article)):

    for x in title_bigrams[i]:

        data_bigrams[str(x)]=article[i]

data_trigrams={}

for i in range(len(article)):

    for x in title_trigrams[i]:

        data_trigrams[str(x)]=article[i]
for x in title_bigrams[1727]:

    pprint({str(x)})

for x in title_trigrams[1727]:

    pprint({str(x)})

print(re.sub("\n|\r", "", str(article[1727])))
sentences=[]

paragraphs = []

words=[]

for item in article[:5]:

    item= re.sub("\n|\r", "", str(item))

    item = re.split('\s{4,}',item)

    for para in item:

        print(para)

        print('\n')

        paragraphs.append(para)

        words.append([x for x in word_tokenize(para) if x.lower() not in stopwords and x not in puncList and x.lower() not in dayMonthList and x.lower() not in wordList])

print(paragraphs[3])

print('\n')

print(words[3])
data_with_bigrams=zip(title_bigrams,article)

with open('data_bigram.json', 'w',encoding='utf-8') as outfile:

    for item in zip(title_bigrams,article):

        json.dump(item, outfile)

        json.dump('\n',outfile)
