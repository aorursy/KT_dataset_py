from fastai import *

from fastai.text import *

from nltk.corpus import stopwords

import re

from bs4 import BeautifulSoup

from functools import partial 

import io 

import os

import sklearn.feature_extraction.text as sklearn_text

import pandas as pd

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

STOPWORDS = set(stopwords.words('english'))



def clean_text(text):

    """

        text: a string

        

        return: modified initial string

    """

    text = BeautifulSoup(text, "lxml").text # HTML decoding

    text = text.lower() # lowercase text

    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text

    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text

    #text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text

    return text
df = pd.read_csv('../input/stack-overflow-user-stories/stack-overflow-data.csv')

df = df[pd.notnull(df['tags'])]

df = df[pd.notnull(df['post'])]

df['post'] = df['post'].apply(clean_text)

df.head()
df.size
df_trn, df_val = train_test_split(df, stratify = df['tags'],  test_size = 0.2, random_state = 12)

df_trn['tags'].value_counts()
labels = list(df['tags'].unique())

data = (TextList.from_df(df_trn, cols='post')

                .split_by_rand_pct(0.2)

                .label_from_df(classes=labels)

                .databunch(bs=48))

data.show_batch()
data.vocab.stoi
len(data.train_dl.x), len(data.valid_dl.x)
def get_term_doc_matrix(label_list, vocab_len):

    j_indices = []

    indptr = []

    values = []

    indptr.append(0)



    for i, doc in enumerate(label_list):

        feature_counter = Counter(doc.data)

        j_indices.extend(feature_counter.keys())

        values.extend(feature_counter.values())

        indptr.append(len(j_indices))



    return scipy.sparse.csr_matrix((values, j_indices, indptr),

                                   shape=(len(indptr) - 1, vocab_len),

                                   dtype=int)







val_term_doc = get_term_doc_matrix(data.valid_dl.x, len(data.vocab.itos))

trn_term_doc = get_term_doc_matrix(data.train_dl.x, len(data.vocab.itos))
x= trn_term_doc

y=data.train_dl.y.items

val_y = data.valid_dl.y.items
m = LogisticRegression(C=0.03, dual=False)

m.fit(x, y)

preds = m.predict(val_term_doc)

(preds==val_y).mean()
m = LogisticRegression(C=0.03, dual=False)

m.fit(trn_term_doc.sign(), y)

preds = m.predict(val_term_doc.sign())

(preds==val_y).mean()
veczr =  CountVectorizer(ngram_range=(1,3), preprocessor=noop, tokenizer=noop, max_features=800000)

docs = data.train_dl.x
train_words = [[docs.vocab.itos[o] for o in doc.data] for doc in data.train_dl.x]

train_ngram_doc = veczr.fit_transform(train_words)

train_ngram_doc
veczr.vocabulary_
valid_words = [[docs.vocab.itos[o] for o in doc.data] for doc in data.valid_dl.x]

val_ngram_doc = veczr.transform(valid_words)

val_ngram_doc
vocab = veczr.get_feature_names()

vocab[100000:100005]
y=data.train_dl.y

valid_labels = data.valid_dl.y.items
m = LogisticRegression(C=0.03, dual=True)

m.fit(train_ngram_doc.sign(), y.items);

preds = m.predict(val_ngram_doc.sign())

(preds.T==valid_labels).mean()
a_list = []

i_list = []

for i in range (1,100):

    m = LogisticRegression(C=i/100, dual=True)

    m.fit(train_ngram_doc.sign(), y.items);

    preds = m.predict(val_ngram_doc.sign())

    a = (preds.T==valid_labels).mean()

    a_list.append(a)

    i_list.append(i)

plt.plot(i_list, a_list)
best_c = i_list[np.argmax(a_list)]/100

best_c
m = LogisticRegression(C=best_c, dual=True)

m.fit(train_ngram_doc.sign(), y.items);

preds = m.predict(val_ngram_doc.sign())

(preds.T==valid_labels).mean()
%matplotlib inline

from sklearn import metrics

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(font_scale=2)

#predictions = model.predict(X_test, batch_size=1000)



LABELS = df['tags'].unique()



confusion_matrix = metrics.confusion_matrix(valid_labels, preds)



plt.figure(figsize=(35, 15))

sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", annot_kws={"size": 20});

plt.title("Confusion matrix", fontsize=20)

plt.ylabel('True label', fontsize=20)

plt.xlabel('Predicted label', fontsize=20)

plt.show()