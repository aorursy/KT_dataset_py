import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack
import gc
import os

# Loading data

train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
train1['lang'] = 'en'

train_es = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-es-cleaned.csv')
train_es['lang'] = 'es'

train_fr = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-fr-cleaned.csv')
train_fr['lang'] = 'fr'

train_pt = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-pt-cleaned.csv')
train_pt['lang'] = 'pt'

train_ru = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-ru-cleaned.csv')
train_ru['lang'] = 'ru'

train_it = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-it-cleaned.csv')
train_it['lang'] = 'it'

train_tr = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-tr-cleaned.csv')
train_tr['lang'] = 'tr'

train = pd.concat([
    
    train1[['comment_text', 'lang', 'toxic']],
    train_es[['comment_text', 'lang', 'toxic']],
    train_tr[['comment_text', 'lang', 'toxic']],
    train_fr[['comment_text', 'lang', 'toxic']],
    train_pt[['comment_text', 'lang', 'toxic']],
    train_ru[['comment_text', 'lang', 'toxic']],
    train_it[['comment_text', 'lang', 'toxic']]
    
]).sample(n=300000, random_state=0).reset_index(drop=True)

del train1, train_es, train_fr, train_pt, train_ru, train_it, train_tr
gc.collect()
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
val  = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
subm = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
train.head()
train.head()
train['comment_text'][0]
train['comment_text'][2]
lens = train.comment_text.str.len()
lens.mean(), lens.std(), lens.max()
lens.hist();
label_cols = ['toxic']
train['none'] = 1-train[label_cols].max(axis=1)
train.describe()
len(train),len(test)
train['comment_text'].fillna("unknown", inplace=True)
test['content'].fillna("unknown", inplace=True)
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
val.head()
n = train.shape[0]

vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )


trn_term_doc = vec.fit_transform(train['comment_text'])
test_term_doc = vec.transform(test['content'])
val_term_doc = vec.transform(val['comment_text'])
trn_term_doc, test_term_doc, val_term_doc
def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)
x = trn_term_doc
test_x = test_term_doc
val_x = val_term_doc
def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=False)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r
label_cols
preds_test = np.zeros((len(test), len(label_cols)))
preds_val = np.zeros((len(val), len(label_cols)))
for i, j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(train[j])
    print('predicting val and test', j)
    preds_test[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
    preds_val[:, i] = m.predict_proba(val_x.multiply(r))[:,1]
from sklearn.metrics import roc_auc_score
print(roc_auc_score(val['toxic'], preds_val))
submid = pd.DataFrame({'id': subm["id"]})
submission_test = pd.concat([submid, pd.DataFrame(preds_test, columns = label_cols)], axis=1)

submission_test.to_csv('nb_svm_strong_lin_baseline_submission_test.csv', index=False)
submid_val = pd.DataFrame({'id': val["id"]})
submission_val = pd.concat([submid_val, pd.DataFrame(preds_val, columns = label_cols)], axis=1)

submission_val.to_csv('nb_svm_strong_lin_baseline_submission_val.csv', index=False)