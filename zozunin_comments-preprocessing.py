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
import pandas as pd

from nltk.tokenize import word_tokenize

import re

import math

import numpy as np

from nltk.corpus import stopwords

stopWords = tuple(stopwords.words('english'))



from nltk.tokenize import TweetTokenizer

twtknr = TweetTokenizer(strip_handles=True, reduce_len=True)

#vs

from nltk.tokenize import word_tokenize

#vs

from nltk.tokenize import TreebankWordTokenizer



from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

#vs

import spacy

nlp = spacy.load('en', disable=['parser', 'ner'])

import textacy



import random

from wordsegment import load, segment

load()



from gensim import utils

from gensim.models import word2vec

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline
train = pd.read_csv('../input/train.csv')
ex = train.iloc[665].comment_text

ex2 = train.iloc[14253].comment_text

ex3 = train.iloc[66059].comment_text
ex
ex2
trb_tok1 = TreebankWordTokenizer().tokenize(ex.lower())

tw_tok1 = twtknr.tokenize(ex.lower())

simp_tok1 = word_tokenize(ex.lower())

trb_tok2 = TreebankWordTokenizer().tokenize(ex2.lower())

tw_tok2 = twtknr.tokenize(ex2.lower())

simp_tok2 = word_tokenize(ex2.lower())

tw_tok1[:10]
simp_tok1[:10]
trb_tok1[:10]
list(set(tw_tok1) ^ set(simp_tok1))
np.array(trb_tok2)
np.array(tw_tok2)
np.array(simp_tok2)
nltk_res = [wordnet_lemmatizer.lemmatize(word) for word in twtknr.tokenize(ex2.lower())]

spacy_res = [token.lemma_ for token in nlp(' '.join(twtknr.tokenize(ex2.lower())))]
' '.join(spacy_res)
' '.join(nltk_res)
nltk_res1 = [wordnet_lemmatizer.lemmatize(word) for word in twtknr.tokenize(ex.lower())]

spacy_res1 = [token.lemma_ for token in nlp(' '.join(twtknr.tokenize(ex.lower())))]
' '.join(spacy_res1)
' '.join(nltk_res1)
train=train.iloc[:1000]
# слова, участвующие в разметке, неважные для общего смысла комментария

drop = [ 'align', 'background', 'background-color', 'bgcolor' 'border', 'border-bottom', 'chapter', 'class', 'colspan', 'font-size', 'lign', 'rowspan', 'scope', 'style', 'valign', 'width' 'padding', 'color', 'text-align', 'cellspacing', 'margin', 'margin-bottom', 'valign', 'vertical-align', 'height', 'border-top']
step = []

for inf, text in enumerate(train.comment_text):

    text = re.sub(r'\{\|.+\n','\n',text)

    for check in re.findall(r'\|[^\|\n]+\|',text):

        if '=' in check and [True for i in drop if i in check]:

            try: text = re.sub(check, '', text)

            except: pass

    for check in re.findall(r'[ \n]![^\!\=\|][^\|\n]+\|',text):

        if '=' in check and [True for i in drop if i in check]:

            text = re.sub(check, '', text)

    for link in re.findall(r'https?://[\w./#?!&-_\(\)]+',text):

        text = re.sub(r'https?://[\w./#?!&-_\(\)]+', '',text)

    step.append(text)
twtokenized = []

for twtext in pd.Series(step):

    twtokenized.append(twtknr.tokenize(twtext.lower()))
def clean():

    cleaned_twtokenized = []

    for text in twtokenized:

        cleaned_text = []

        for t in text:

            if not re.findall(r'^[^a-zA-Z]$',t) and t!='...' and t not in drop:

                if re.findall(r'^\w+$',t):

                    if not re.findall(r'[^a-z_0-9]+',t):

                        if not re.findall(r'^\d+[^a-z]+',t):

                            cleaned_text.append(t)

                else:

                    if not re.findall(r'[\w]+\.[\w]+',t) and not re.findall(r'^\d+[^a-z]',t):

                        cleaned_text.append(t)

        fin = ' '.join(cleaned_text)

        fin = re.sub('#','',fin)

        fin = re.sub('::','',fin)

        fin = re.sub('_',' ',fin)

        fin = re.sub('wikipedia','',fin)

        cleaned_twtokenized.append(fin)

    return cleaned_twtokenized
cleaned = clean()
ex3
segment(ex3)
re_cleaned = [' '.join(segment(i)) for i in cleaned]
def fin_clean(stopwords):

    lem_com = []

    for i, comment in enumerate(re_cleaned):

        doc = nlp(comment)

        lem_com.append(" ".join([token.lemma_ for token in doc if token.lemma_ not in stopwords]))

        if i%10000 == 0:

            print(i, 'out of', len(re_cleaned))

    return lem_com

cl_lem = pd.Series(np.array(fin_clean(stopWords)))
# финальное удаление остаточных односимвольных токенов, а также числовых данных

rechecked = []

for comment in cl_lem:

    temp = comment.split(' ')

    temp_ = []

    for i in temp:

        if len(i) != 1 or not i.isdigit():

            temp_.append(i)

    rechecked.append(' '.join(temp_))
data = pd.concat([train, pd.Series(np.array(rechecked))], axis=1)

data = data.drop(['id','comment_text'], axis=1)

data.rename(columns = {0:'cleaned'},inplace=True)
data.cleaned.head()