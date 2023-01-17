import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df_train = pd.read_csv('../input/train.csv')

df_train.head()
df_test = pd.read_csv('../input/test.csv')

df_test.head()
print('Total number of question pairs for testing: {}'.format(len(df_test)))
import re, math, collections

from nltk.corpus import stopwords



stws = set(stopwords.words("english"))

 

def tokenize(_str):

  tokens = collections.defaultdict(lambda: 0.)

  for m in re.finditer(r"(\w+)", _str, re.UNICODE):

    m = m.group(1).lower()

    if len(m) < 2: continue

    if m in stws: continue

    tokens[m] += 1  

  return tokens



def kldivergence(_s, _t):

    if (len(_s) == 0):

        return 1e33

 

    if (len(_t) == 0):

        return 1e33

 

    ssum = 0. + sum(_s.values())

    slen = len(_s)

 

    tsum = 0. + sum(_t.values())

    tlen = len(_t)

 

    vocabdiff = set(_s.keys()).difference(set(_t.keys()))

    lenvocabdiff = len(vocabdiff)

 

    """ epsilon """

    epsilon = min(min(_s.values())/ssum, min(_t.values())/tsum) * 0.001

 

    """ gamma """

    gamma = 1 - lenvocabdiff * epsilon

 

    """ Check if distribution probabilities sum to 1"""

    sc = sum([v/ssum for v in _s.values()])

    st = sum([v/tsum for v in _t.values()])

 

    if sc < 9e-6:

        sys.exit(2)

    if st < 9e-6:

        sys.exit(2)

 

    div = 0.

    for t, v in _s.items():

        pts = v / ssum

 

        ptt = epsilon

        if t in _t:

            ptt = gamma * (_t[t] / tsum)

 

        ckl = (pts - ptt) * math.log(pts / ptt)

 

        div +=  ckl

 

    return div



q1 = "What are the how best books of all time?"

q2 = "What are some of the military history books of all time?"

 

print("KL-divergence between q1 and q2:", kldivergence(tokenize(q1), tokenize(q2)))

print("KL-divergence between d2 and d1:", kldivergence(tokenize(q2), tokenize(q1)))

    
def kldistance(q1, q2):

  q1t = tokenize(q1)

  q2t = tokenize(q2)

  q1q2div = kldivergence(q1t, q2t)

  q2q1div = kldivergence(q2t, q1t)

  divs = q1q2div + q2q1div

  if divs == 0: return 0

  return (2 * q1q2div * q2q1div) / divs



print(kldistance(q1, q2))
def kldr(row):

  return kldistance(str(row['question1']), str(row['question2']))    



train_kld = df_train.apply(kldr, axis=1, raw=True)

test_kld = df_test.apply(kldr, axis=1, raw=True)



# First we create our training and testing data

x_train = pd.DataFrame()

x_test = pd.DataFrame()

x_train['kld'] = train_kld

x_test['kld'] = test_kld



y_train = df_train['is_duplicate'].values
pos_train = x_train[y_train == 1]

neg_train = x_train[y_train == 0]



# Now we oversample the negative class

# There is likely a much more elegant way to do this...

p = 0.165

scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1

while scale > 1:

    neg_train = pd.concat([neg_train, neg_train])

    scale -=1

neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])

print(len(pos_train) / (len(pos_train) + len(neg_train)))



x_train = pd.concat([pos_train, neg_train])

y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()

del pos_train, neg_train
# Finally, we split some of the data off for validation

from sklearn.cross_validation import train_test_split



x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
import xgboost as xgb



# Set our parameters for xgboost

params = {}

params['objective'] = 'binary:logistic'

params['eval_metric'] = 'logloss'

params['eta'] = 0.02

params['max_depth'] = 4



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
d_test = xgb.DMatrix(x_test)

p_test = bst.predict(d_test)



sub = pd.DataFrame()

sub['test_id'] = df_test['test_id']

sub['is_duplicate'] = p_test

sub.to_csv('kld_xgb.csv', index=False)

sub.head()