import tensorflow as tf

print(tf.__version__)
import numpy as np

import pandas as pd

import json



def read_lines_m(path, max_limit=4000):

    rlm = []; ml = max_limit

    for l in open(path, 'r'):

        rlm.append(json.loads(l))

        ml -= 1

        if ml <= 0: break

    return pd.DataFrame(rlm)



p = '../input/tensorflow2-question-answering/'

train = read_lines_m(p + 'simplified-nq-train.jsonl')

train['D'] = [t[0]['long_answer']['start_token'] for t in train.annotations]

train = train[train['D']>-1].reset_index(drop=True)

test = read_lines_m(p + 'simplified-nq-test.jsonl').reset_index(drop=True)

sub = pd.read_csv(p + 'sample_submission.csv')

train.shape, test.shape, sub.shape
i =99

print('URL:', train.document_url[i])

print(train.question_text[i])

print(train.long_answer_candidates[i][0])

print(' '.join(train.document_text[i].split()[train.annotations[i][0]['long_answer']['start_token'] : train.annotations[i][0]['long_answer']['end_token']]))

if len(train.annotations[i][0]['short_answers']) > 0:

    print(' '.join(train.document_text[i].split()[train.annotations[i][0]['short_answers'][0]['start_token'] : train.annotations[i][0]['short_answers'][0]['end_token']]))
la=[t[0]['long_answer']['end_token'] - t[0]['long_answer']['start_token'] for t in train.annotations]

sa=[t[0]['short_answers'][0]['end_token'] - t[0]['short_answers'][0]['start_token'] for t in train.annotations if len(t[0]['short_answers'])>0]

np.median(la), np.median(sa)
from bs4 import BeautifulSoup as b

from nltk.corpus import stopwords

import random, nltk



def qa_word_match(q,a):

    q = q.lower().split()

    q = [q1 for q1 in q if q1 not in list(set(stopwords.words('english')))]

    tm = 0

    a2 = a[0]

    for a1 in a:

        m = np.sum([1 for w in a1.lower().split() if w in q])

        if m > tm:

            tm = int(m)

            a2 = str(a1)

    return a2



result = []

for i in range(len(test.example_id)):

    s = b(test.document_text[i], 'html.parser')

    p = [p.get_text() for p in s.find_all('p', text=True) if len(p.get_text()) > 50]

    if len(p)>0:

        a = qa_word_match(test.question_text[i], p)

        r = test.document_text[i].find(a)

        r = len(test.document_text[i][:r].split()) - 1

        long_answer= ''.join([str(r),':', str(r + len(p[0].split()) + 2)])

    else:

        try:

            r = random.randrange(390, len(test.document_text[i].split()))

        except:

            r=7

        long_answer= ''.join([str(r),':', str(r + 114)])



    if len([q for q in ['am', 'are', 'can', 'could', 'did', 'do', 'does', 'has', 'have', 'is', 'may', 'should', 'was', 'were', 'will'] if q in test.question_text[i].lower().split()])>0:

        short_answer=random.choice(['YES','NO'])

    else:

        r = random.randrange(r, r + 114)

        short_answer=''.join([str(r),':', str(r + 2)])

    result.append([test.example_id[i] + '_long', long_answer])

    result.append([test.example_id[i] + '_short', short_answer])

pd.DataFrame(result,columns=['example_id', 'PredictionString']).to_csv('submission.csv', index=False)