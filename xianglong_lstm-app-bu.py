# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import gc
import os
start = time.time()
os.chdir('/kaggle/input/home-credit-default-risk')
#os.chdir('/Users/xianglongtan/Desktop/kaggle')
#print(os.getcwd())
#print(os.listdir())
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_activity = 'all'
# Any results you write to the current directory are saved as output.
app_train = pd.read_csv('application_train.csv')
#app_train = pd.read_csv('application_train.csv')
#app_train.head()
app_test = pd.read_csv('application_test.csv')
#app_test = pd.read_csv('application_test.csv')
#app_test.head()
os.chdir('../imputed')
tnt = pd.read_csv('lstm_app_bu.csv')
os.chdir('/kaggle/working')
import json
import numpy as np
def token(X):
    record = [status for status in X.STATUS.values]
    status = [json.loads(status) for status in record]
    sta = []
    for st in status:
        sta.extend([s for s in st])
    return np.matrix(sta).T
label_status = token(tnt)
from sklearn.preprocessing import LabelEncoder
labelenc = LabelEncoder()
int_encode = labelenc.fit(label_status)
status_enc = int_encode.transform(label_status)
tnt['STATUS'] = tnt['STATUS'].map(lambda x: json.loads(x))
tnt['STATUS_INT'] = tnt['STATUS'].apply(lambda x: int_encode.transform(x))
tnt = tnt.drop('STATUS',axis=1)
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
cols = [x for x in tnt.columns if tnt[x].dtype == 'float64' or tnt[x].dtype == 'int64']
tnt[cols] = normalize(tnt[cols],axis=0)
tnt[cols] = scale(tnt[cols],axis=0)
length = len(tnt.STATUS_INT.values[0])
train_length=307511
train_X = tnt.loc[:train_length]
test_X = tnt.loc[train_length:]
train_Y = app_train['TARGET']
del app_train
gc.collect()
train_X = train_X.drop(['Unnamed: 0','SK_ID_CURR'],axis=1)
train_X.drop([train_X.shape[0]-1],axis=0,inplace=True)
test_X.drop('Unnamed: 0',axis=1,inplace=True)
test_X.set_index('SK_ID_CURR',inplace=True)
import tensorflow as tf
def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

reset_graph()

# Hyper Param
N_CONT = len([x for x in train_X.columns if train_X[x].dtype == 'float64'])
BATCH_SIZE = 400
n_classes = 2
length_record = len(train_X.STATUS_INT.values[0])# time steps
num_status = max(status_enc)
CELL_SIZE = 50
KEEP_PROB = 0.7
LR = 0.001
ITERATION = 3000
seed = 0
EMB_SIZE = 400

class DataIter():
    def __init__(self, X,Y,N_CONT,seed):
        self.X = X
        self.Y = Y
        self.N_CONT = N_CONT
        self.size = len(self.X)
        self.df = pd.concat([X,Y],axis=1)
        self.pos = self.df.loc[self.Y == 1]
        self.neg = self.df.loc[self.Y == 0]
        self.seed = seed
    def next_batch(self,n):
        pos_sample = self.pos.sample(round(n/2), replace=True,random_state=self.seed)
        neg_sample = self.neg.sample(n-round(n/2), replace=True,random_state=self.seed)
        res = pd.concat([neg_sample, pos_sample],axis=0)
        status = np.zeros([n,length_record],dtype=np.int32)
        for i, status_i in enumerate(status):
            status_i = res.iloc[:,N_CONT:-1].values[i]
        return res.iloc[:,:N_CONT].values,status,res.iloc[:,-1].values   
    
    
    
X_cont = tf.placeholder(tf.float64, shape=(BATCH_SIZE, N_CONT), name='X_CONT')
y = tf.placeholder(tf.int64, shape=(BATCH_SIZE), name='TARGET')

# Add embeddings
X = tf.identity(X_cont)
X = tf.reshape(X, [BATCH_SIZE, 1, N_CONT])
status = tf.placeholder(tf.int32,[BATCH_SIZE,length_record])
seqlen = tf.constant(length_record)
seqlen = tf.reshape(seqlen, [1])
seqlen = tf.tile(seqlen, [BATCH_SIZE])
embedding = tf.Variable(tf.random_uniform([num_status, EMB_SIZE], -1.0, 1.0))
embedded_x = tf.nn.embedding_lookup(embedding, status)
X = tf.tile(X,[1,length_record,1])
X = tf.cast(X,tf.float32)
X = tf.concat([embedded_x, X], axis=2)

# RNN
with tf.name_scope('RNN'):
    # input layer
    rnn_input = tf.layers.dense(X,units = CELL_SIZE)
    
    # rnn
    cell = tf.nn.rnn_cell.GRUCell(CELL_SIZE)
    init_state = tf.get_variable('init_state', [1,CELL_SIZE],
                                initializer = tf.constant_initializer(0.0))
    init_state = tf.tile(init_state, [BATCH_SIZE, 1])
    rnn_output, final_state = tf.nn.dynamic_rnn(cell, rnn_input, 
                                                sequence_length = seqlen,
                                                initial_state = init_state)
    
    # dropout
    rnn_output = tf.nn.dropout(rnn_output, KEEP_PROB)
    
    # select last output
    last_rnn_output = tf.gather_nd(rnn_output, tf.stack([tf.range(BATCH_SIZE), seqlen-1], axis=1))
    last_rnn_output = tf.layers.dense(last_rnn_output, units = 1.5*CELL_SIZE)
    last_rnn_output = tf.layers.dense(last_rnn_output, units = 0.5*CELL_SIZE)
    last_rnn_output = tf.layers.dense(last_rnn_output, units = 0.1*CELL_SIZE)
    
# Cost
with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [0.1*CELL_SIZE, n_classes])
    b = tf.get_variable('b', [n_classes], initializer = tf.constant_initializer(0.0))
logits = tf.matmul(last_rnn_output, W)+b
preds = tf.nn.softmax(logits)
prediction = tf.cast(tf.argmax(preds,1), tf.int32)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = preds))
precision, precision_op = tf.metrics.precision(y,prediction)
recall, recall_op = tf.metrics.recall(y,prediction)
f1score = 2*precision*recall/(precision+recall)
auc,auc_op = tf.metrics.auc(y,preds[:,1])
train_step = tf.train.AdamOptimizer(LR).minimize(loss)


# session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    tr = DataIter(train_X,train_Y,N_CONT,seed)
    for i in range(ITERATION):
        x_cont, x_sta, target = tr.next_batch(BATCH_SIZE)
        feed = {X_cont:x_cont, status:x_sta, y:target}
        sess.run(train_step, feed_dict=feed)
        if i%200 == 0:
            _,prec = sess.run([precision,precision_op], feed_dict=feed)
            _,rec = sess.run([recall,recall_op], feed_dict=feed)
            auc_score,auc_update = sess.run([auc,auc_op],feed_dict=feed)
            f1s = sess.run(f1score, feed_dict=feed)
            los = sess.run(loss, feed_dict=feed)
            print('losss after',i,'round',los)
            print('precision after',i,'round',prec)
            print('recall after',i,'round',rec)
            print('F1 score after',i,'round:',f1s)
            print('auc after',i,'round',auc_update)
            #print('\n----------------------------------\n')
            #print('logits:\n',sess.run(logits,feed_dict=feed)[0:10])
            #print('preds:\n',sess.run(preds, feed_dict=feed)[0:10])
            #print('prediction:\n',sess.run(prediction, feed_dict=feed)[0:10])
            #print('y:\n',sess.run(y,feed_dict=feed)[0:10])
            print('\n----------------------------------\n')
    cursor = 0
    while cursor <= len(test_X):
        if cursor+BATCH_SIZE <= len(test_X):
            cont_test = test_X.iloc[cursor:cursor+BATCH_SIZE, :N_CONT]
            status_test = test_X.iloc[cursor:cursor+BATCH_SIZE, N_CONT:]
        else:
            cont_test = pd.concat([test_X.iloc[cursor:len(test_X), :N_CONT],
                                   test_X.iloc[0:(BATCH_SIZE-len(test_X.iloc[cursor:len(test_X), :N_CONT])),:N_CONT]],axis=0)
            print(cont_test.shape)
            status_test = pd.concat([test_X.iloc[cursor:len(test_X), N_CONT:],
                                   test_X.iloc[0:(BATCH_SIZE-len(test_X.iloc[cursor:len(test_X), N_CONT:])), :N_CONT]],axis=0)
        sta_te = np.zeros([BATCH_SIZE,length_record],dtype=np.int32)
        for i,sta_i in enumerate(sta_te):
            sta_i = status_test.values[i]
        results = sess.run(preds, feed_dict={X_cont:cont_test, status:sta_te})
        if cursor == 0:
            prediction_test = pd.DataFrame(data=results, columns=['0','TARGET'])
        else:
            prediction_test = pd.concat([prediction_test, pd.DataFrame(data=results,columns=['0','TARGET'])])
        cursor += BATCH_SIZE
prediction_test[['TARGET']].head(10)
result = app_test[['SK_ID_CURR']].merge(prediction_test[['TARGET']].iloc[:len(app_test)].reset_index(drop=True),left_index=True,right_index=True).set_index('SK_ID_CURR')
result.to_csv('submission_lstm.csv')
'''
bb = pd.read_csv('bureau_balance.csv')
b = pd.read_csv('bureau.csv')
os.chdir('../imputed')
#print(os.listdir())
tnt = pd.read_csv('train_and_test_imputed.csv')
#os.chdir('/Users/xianglongtan/Desktop/kaggle/submission')
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',200)
col = ['SK_ID_CURR','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY',
       'EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']
tnt = tnt[col]
def fn(car, realty):
    if car == 'N' and realty == 'N':
        return int(0)
    elif car == 'Y' and realty == 'N':
        return int(1)
    elif car == 'N' and realty == 'Y':
        return int(2)
    elif car == 'Y' and realty == 'Y':
        return int(3)
tnt['OWN_CAR_REALTY'] = tnt.apply(lambda x: fn(x['FLAG_OWN_CAR'],x['FLAG_OWN_REALTY']), axis=1)
tnt = tnt.drop(['FLAG_OWN_CAR','FLAG_OWN_REALTY'],axis=1)
b2 = b.merge(bb, left_on='SK_ID_BUREAU',right_on='SK_ID_BUREAU',how='left',suffixes=['_b','_bb'])
import gc
gc.enable()
del bb
del b
gc.collect()
tnt_b2 = tnt.merge(b2[['SK_ID_CURR','MONTHS_BALANCE','STATUS']], left_on='SK_ID_CURR',right_on='SK_ID_CURR',how='left',suffixes=['_tnt','_b2'])
map_dict = {'C':1000000,'X':10000000,'0':1,'1':10,'2':100,'3':1000,'4':10000,'5':100000}
tnt_b2['STATUS'] = tnt_b2['STATUS'].map(map_dict)
agg_fun={'STATUS':'sum'}
status = tnt_b2.groupby(['SK_ID_CURR','MONTHS_BALANCE']).agg(agg_fun)
status = status.pivot_table(values='STATUS',index='SK_ID_CURR',columns='MONTHS_BALANCE')
agg_fun = {'CNT_CHILDREN':'mean',
        'AMT_INCOME_TOTAL':'mean',
        'AMT_ANNUITY':'mean',
        'EXT_SOURCE_1':'mean',
        'EXT_SOURCE_2':'mean',
        'EXT_SOURCE_3':'mean',
        'OWN_CAR_REALTY':'mean'}
other = tnt_b2.groupby('SK_ID_CURR').agg(agg_fun)
tnt_b2 = pd.concat([other,status],axis=1).reset_index()
os.chdir('/kaggle/input/home-credit-default-risk')
app_train = pd.read_csv('application_train.csv')
train_ix = app_train[['SK_ID_CURR','TARGET']]
train_X = tnt_b2[tnt_b2[['SK_ID_CURR']].isin(train_ix['SK_ID_CURR'].values).values]
train_y = train_ix.copy()
app_test = pd.read_csv('application_test.csv')
test_ix = app_test[['SK_ID_CURR']]
test_X = tnt_b2[tnt_b2[['SK_ID_CURR']].isin(test_ix['SK_ID_CURR'].values).values]
del tnt,b2
del app_train
del train_ix
del app_test, test_ix
gc.collect()
train_X = train_X.fillna(0)
test_X = test_X.fillna(0)
cols = [x for x in train_X.columns if str(x).startswith('-') or str(x).startswith('0')]
train_X['STATUS'] = train_X[cols].values.tolist()
train_X = train_X.drop(cols,axis=1)
test_X['STATUS'] = test_X[cols].values.tolist()
test_X = test_X.drop(cols,axis=1)
tnt = pd.concat([train_X,test_X],axis=0)
os.chdir('/kaggle/working')
os.getcwd()
#tnt.to_csv('lstm_app_bu.csv')
end = time.time()
print(end-start)
'''
