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
import tensorflow as tf 
import matplotlib.pyplot as plt 
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
from nltk.corpus import stopwords
import re,string,unicodedata
from nltk.stem.porter import PorterStemmer
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)
df = pd.read_csv("/kaggle/input/million-headlines/abcnews-date-text.csv")
df.head(3)
df.shape
df.publish_date.value_counts().tail(30)
import spacy 
nlp = spacy.load('en_core_web_lg')
def text_entity(text):
    doc = nlp(text)
    for ent in doc.ents:
        print(f'Entity: {ent}, Label: {ent.label_}, {spacy.explain(ent.label_)}')
text_entity(df['headline_text'][10])
first = df['headline_text'][50]
doc = nlp(first)
spacy.displacy.render(doc, style='ent',jupyter=True)
first = df['headline_text'][2000]
doc = nlp(first)
spacy.displacy.render(doc, style='ent',jupyter=True)
first = df['headline_text'][8000]
doc = nlp(first)
spacy.displacy.render(doc, style='ent',jupyter=True)
txt = df['headline_text'][2000]
doc = nlp(txt)
spacy.displacy.render(doc, style='ent', jupyter=True)

for idx, sentence in enumerate(doc.sents):
    for noun in sentence.noun_chunks:
        print(f"sentence {idx+1} has noun chunk '{noun}'")
txt = df['headline_text'][5600]
doc = nlp(txt)
spacy.displacy.render(doc, style='ent', jupyter=True)

for token in doc:
    print(token, token.pos_)
headline_length=df['headline_text'].str.len()
sns.distplot(headline_length)
plt.show()
headline_length=df['headline_text'].str.len()
plt.hist(headline_length)
plt.show()
df_ = df['headline_text'].str.cat(sep=' ')

max_length = 1000000-1
df_ =  df_[:max_length]

import re
url_reg  = r'[a-z]*[:.]+\S+'
df_   = re.sub(url_reg, '', df_)
noise_reg = r'\&amp'
df_   = re.sub(noise_reg, '', df_)
doc = nlp(df_)
items_of_interest = list(doc.noun_chunks)
items_of_interest = [str(x) for x in items_of_interest]
df_nouns = pd.DataFrame(items_of_interest, columns=["Corona"])
plt.figure(figsize=(5,4))
sns.countplot(y="Corona",
             data=df_nouns,
             order=df_nouns["Corona"].value_counts().iloc[:10].index)
plt.show()
distri = df['headline_text'][2000]
doc = nlp(distri)
options = {'compact': True, 'bg': '#09a3d5',
           'color': 'white', 'font': 'Trebuchet MS'}
spacy.displacy.render(doc, jupyter=True, style='dep', options=options)
distri = df['headline_text'][7000]
doc = nlp(distri)
options = {'compact': True, 'bg': '#09a3d5',
           'color': 'white', 'font': 'Trebuchet MS'}
spacy.displacy.render(doc, jupyter=True, style='dep', options=options)
for token in doc:
    print(f"token: {token.text},\t dep: {token.dep_},\t head: {token.head.text},\t pos: {token.head.pos_},\
    ,\t children: {[child for child in token.children]}")
stemmer = PorterStemmer()
def stem_text(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            word = stemmer.stem(i.strip())
            final_text.append(word)
    return " ".join(final_text)    
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(df.headline_text))
plt.imshow(wc , interpolation = 'bilinear')