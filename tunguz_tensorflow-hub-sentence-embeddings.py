import os
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
import gc

import lightgbm as lgb

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression


import matplotlib.pyplot as plt
import seaborn as sns

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow_hub as hub

%matplotlib inline

train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')
test_text = test['question_text'].values.tolist()
start_time = time()
embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
time() - start_time
embeddings = embed(test_text)
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    test_embeddings = session.run(embeddings)
test_embeddings.shape
train_text = train['question_text'].values.tolist()
len(train_text)/25
train_text = [train_text[i:i + 52250] for i in range(0, len(train_text), 52250)]

len(train_text)

embeddings_train = []
for i in tqdm(range(25)):
    embeddings = embed(train_text[i])
    embeddings_train.append(embeddings)
train_embeddings_all = []
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    for i in tqdm(range(25)):
        train_embeddings = session.run(embeddings_train[i])
        train_embeddings_all.append(train_embeddings)
del train_text, test_text
gc.collect()
train_embeddings_all = np.vstack(train_embeddings_all)
train_embeddings_all.shape
train_target = train['target'].values
del train, test
gc.collect()
kf = KFold(n_splits=5, shuffle=True, random_state=43)
test_pred_tf = 0
oof_pred_tf = np.zeros([train_embeddings_all.shape[0],])

for i, (train_index, val_index) in tqdm(enumerate(kf.split(train_embeddings_all))):
    x_train, x_val = train_embeddings_all[train_index,:], train_embeddings_all[val_index,:]
    y_train, y_val = train_target[train_index], train_target[val_index]
    classifier = LogisticRegression(class_weight = "balanced", C=0.5, solver='sag')
    classifier.fit(x_train, y_train)
    val_preds = classifier.predict_proba(x_val)[:,1]
    preds = classifier.predict_proba(test_embeddings)[:,1]
    test_pred_tf += 0.2*preds
    oof_pred_tf[val_index] = val_preds
np.save('train_embeddings_all', train_embeddings_all)
np.save('test_embeddings', test_embeddings)
np.save('train_target', train_target)
pred_train = (oof_pred_tf > 0.8).astype(np.int)
f1_score(train_target, pred_train)
test = pd.read_csv('../input/test.csv').fillna(' ')
pred_test = (test_pred_tf> 0.8).astype(np.int)
submission = pd.DataFrame.from_dict({'qid': test['qid']})
submission['prediction'] = pred_test
submission.to_csv('submission.csv', index=False)
