import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)
train.describe()
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])
trn_term_doc, test_term_doc
def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)
x = trn_term_doc
test_x = test_term_doc
def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r
x = trn_term_doc
j = label_cols[0]
j
y = train[j]
y = y.values
p = x[y==1].sum(0)+1
q = x[y==0].sum(0)+1
p_n_bk = p.sum()
q_n_bk = q.sum()
r_bk = np.log( (p/p_n_bk) / (q/q_n_bk))
np.allclose(r_bk, np.log(p/q) + np.log(q_n_bk/p_n_bk))
p_n_jh = (y==1).sum()+1
q_n_jh = (y==0).sum()+1
r_jh = np.log( (p/p_n_jh) / (q/q_n_jh))
np.allclose(r_jh, np.log(p/q) + np.log(q_n_jh/p_n_jh))
cnst = np.log(q_n_jh/p_n_jh) - np.log(q_n_bk/p_n_bk)
cnst
np.allclose(r_jh, r_bk + cnst)
xm = x.multiply(r_bk) 
xmjh = x.multiply(r_jh)
np.allclose(xmjh.tocsr()[0].todense(), 
            (xm + x*cnst)[0].todense())
