import pandas as pd

from pandas import DataFrame

import nltk

from tqdm import tqdm

from contextlib import contextmanager

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import StratifiedKFold

from sklearn.naive_bayes import MultinomialNB

from nltk.tokenize import word_tokenize

import time

import numpy as np

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

from nltk.stem.snowball import SnowballStemmer

from sklearn.model_selection import train_test_split
def tokenize(raw):

    return [w.lower() for w in word_tokenize(raw) if w.isalpha()]



class StemmedTfidfVectorizer(TfidfVectorizer):

    en_stemmer = SnowballStemmer('english')

    

    def build_analyzer(self):

        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()

        return lambda doc: (StemmedTfidfVectorizer.en_stemmer.stem(w) for w in analyzer(doc))
@contextmanager

def timer(task_name="timer"):

    # a timer cm from https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s

    print("----{} started".format(task_name))

    t0 = time.time()

    yield

    print("----{} done in {:.0f} seconds".format(task_name, time.time() - t0))
with timer("reading_data"):

    train = pd.read_csv("../input/train.csv")

    test_df = pd.read_csv('../input/test.csv')
## split to train and val

train_df, val_df = train_test_split(train, test_size=0.1, random_state=2018)
tfidf = StemmedTfidfVectorizer(

    tokenizer=tokenize, 

    analyzer="word", 

    stop_words='english', 

    ngram_range=(1,1), 

    min_df=3    # limit of minimum number of counts: 3

)



with timer('tfidf train'):

    txt_all = pd.concat([train.question_text, test_df.question_text])

    tfidf.fit(txt_all)

    

with timer('construct training and validation dataset'):

    train_X = tfidf.transform(train_df.question_text)

    val_X = tfidf.transform(val_df.question_text)

    

with timer('transforming the test set'):

    test_X = tfidf.transform(test_df['question_text'])

    

## Get the target values

train_y = train_df['target'].values

val_y = val_df['target'].values
clf = MultinomialNB().fit(train_X, train_y)

y_val_pred = clf.predict_proba(val_X)
# threshold search

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score at threshold {0} is {1}".format(thresh, f1_score(val_y, (y_val_pred[:,1] > thresh).astype(int))))
y_pred = clf.predict_proba(test_X)

pred_test_y = (y_pred[:, 1] > 0.22).astype(int)

out_df = pd.DataFrame({"qid":test_df["qid"].values})

out_df['prediction'] = pred_test_y

out_df.to_csv("submission.csv", index=False)