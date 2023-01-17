# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.getcwd())
#os.chdir('/Users/xianglongtan/Desktop/kaggle')
print(os.listdir("../input"))
#print(os.listdir())
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_activity = 'all'
# Any results you write to the current directory are saved as output.
app_train = pd.read_csv('../input/application_train.csv')
#app_train = pd.read_csv('application_train.csv')
#app_train.head()
app_test = pd.read_csv('../input/application_test.csv')
#app_test = pd.read_csv('application_test.csv')
#app_test.head()
train_Y = app_train['TARGET']
train_X = app_train.drop('TARGET',axis=1)
test_X = app_test
print(train_Y.shape)
print(train_X.shape)
print(test_X.shape)
# training set
#train_Y.isnull().sum() # no missing value
train_X_nonan = train_X.loc[:,train_X.isnull().sum() == 0] # no nan columns
train_X_nan = train_X.loc[:,train_X.isnull().sum()>0] # columns that have nan
num_nan_train = train_X_nan.isnull().sum()
num_nan_train = pd.DataFrame(num_nan_train)
num_nan_train = num_nan_train.reset_index()
num_nan_train.columns = ['columns','number']
#num_nan_train
# test set
test_X_nonan = test_X.loc[:,test_X.isnull().sum() == 0] # no nan columns
test_X_nan = test_X.loc[:,test_X.isnull().sum()>0] # columns that have nan
num_nan_test = test_X_nan.isnull().sum()
num_nan_test = pd.DataFrame(num_nan_test)
num_nan_test = num_nan_test.reset_index()
num_nan_test.columns = ['columns','number']
#num_nan_test
# test sets has 64 features with missing value while trainging set has 67 features
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)
df = num_nan_train.set_index('columns').join(num_nan_test.set_index('columns'), how='left', lsuffix='_train', rsuffix='_test')
train_X_nan = train_X[list(df.index)]
train_X_nan.head(10)
less_10k_nan_train = num_nan_train[num_nan_train.number <= 10000] 
train_X_useful = train_X[less_10k_nan_train['columns']]# columns that have less than 10k nan
less_10k_nan_test = num_nan_test[num_nan_test.number <= 8000]
test_X_useful = test_X[less_10k_nan_test['columns']]
train_X_useful.isnull().sum()
test_X_useful.isnull().sum()
# select columns that both not have missing values in training and test set
nonan_columns =  train_X_nonan.columns.intersection(test_X_nonan.columns).drop('NAME_EDUCATION_TYPE')
train_and_test = pd.concat([train_X_nonan[nonan_columns], test_X_nonan[nonan_columns]], axis=0)
train_and_test_object = train_and_test.loc[:,train_and_test.dtypes==object]
object_col = train_and_test_object.columns
train_X_nonan_obj = train_X_nonan[object_col]
test_X_nonan_obj = test_X_nonan[object_col]
train_X_nonan_dummies = pd.get_dummies(train_X_nonan_obj)
test_X_nonan_dummies = pd.get_dummies(test_X_nonan_obj)
train_X_nonan_dummies,test_X_nonan_dummies = train_X_nonan_dummies.align(test_X_nonan_dummies, join='inner', axis=1)
# education is order categorical features
def encode_edu(x):
    if x == 'Secondary / secondary special':
        return np.float(1)
    elif x == 'Higher education':
        return np.float(3)
    elif x == 'Incomplete higher':
        return np.float(2)
    elif x == 'Lower secondary':
        return np.float(0)
    else:
        return np.float(4)
education = test_X_nonan.NAME_EDUCATION_TYPE.map(lambda x: encode_edu(x))
test_X_nonan_dummies = pd.concat([test_X_nonan_dummies, education],axis=1)
education = train_X_nonan.NAME_EDUCATION_TYPE.map(lambda x: encode_edu(x))
train_X_nonan_dummies = pd.concat([train_X_nonan_dummies, education],axis=1)
nonan_num_col = train_and_test.loc[:,train_and_test.dtypes!=object].columns
test_X_nonan_dummies = pd.concat([test_X_nonan_dummies, test_X_nonan[nonan_num_col]],axis=1)
train_X_nonan_dummies = pd.concat([train_X_nonan_dummies, train_X_nonan[nonan_num_col]],axis=1)

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost.sklearn import XGBClassifier
import time
from plotnine import *
Y = train_Y
X = train_X_nonan_dummies.drop('SK_ID_CURR',axis=1)
seed = 4
test_X = test_X_nonan_dummies
test_ID = pd.DataFrame(test_X['SK_ID_CURR'])
test_X = test_X_nonan_dummies.drop('SK_ID_CURR',axis=1)
X_train,X_val,y_train,y_val = train_test_split(X,Y,random_state=seed,stratify=Y)
#y_train = pd.DataFrame(y_train)
#y_val = pd.DataFrame(y_val)
#(ggplot(y_train)+geom_bar(aes(x='TARGET')))
#(ggplot(y_val)+geom_bar(aes(x='TARGET')))
# KNN
flag = 0
if flag == 0:
    pass
else:
    start = time.time()
    model = KNeighborsClassifier()
    #model.fit(X_train, y_train)
    #pred_train = model.predict(X_train)
    #pred_val = model.predict(X_val)
    model.fit(X,Y)
    end  = time.time()
    #print('F1 score of training set:',f1_score(pred_train, y_train, average='weighted'))
    #print('F1 score of test set:',f1_score(pred_val, y_val, average='weighted'))
    print('Done! Time spent:',end-start)
    pred_test = pd.DataFrame(model.predict_proba(test_X)).loc[:,1]
    result = pd.concat([test_ID,pred_test],axis=1)
    result.columns = ['SK_ID_CURR','TARGET']
    result = result.set_index('SK_ID_CURR')
    result.to_csv('only_nonan_knn1.csv')
# SVM
flag = 0
if flag == 0:
    pass
else:
    start = time.time()
    model = SVC()
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    end  = time.time()
    #print('F1 score of training set:',f1_score(pred_train, y_train, average='weighted'))
    #print('F1 score of test set:',f1_score(pred_val, y_val, average='weighted'))
    print('Done! Time spent:',end-start)
    pred_test = pd.DataFrame(model.predict_proba(test_X)).loc[:,1]
    result = pd.concat([test_ID,pred_test],axis=1)
    result.columns = ['SK_ID_CURR','TARGET']
    result = result.set_index('SK_ID_CURR')
    result.to_csv('only_nonan_svm1.csv')



# Logistic Regression
flag = 0
if flag == 0:
    pass
else:
    start = time.time()
    model = LogisticRegression()
    #model.fit(X_train, y_train)
    #pred_train = model.predict(X_train)
    #pred_val = model.predict(X_val)
    model.fit(X,Y)
    end  = time.time()
    #print('F1 score of training set:',f1_score(pred_train, y_train, average='weighted'))
    #print('F1 score of test set:',f1_score(pred_val, y_val, average='weighted'))
    print('Done! Time spent:',end-start)
    pred_test = pd.DataFrame(model.predict_proba(test_X)).loc[:,1]
    result = pd.concat([test_ID,pred_test],axis=1)
    result.columns = ['SK_ID_CURR','TARGET']
    result = result.set_index('SK_ID_CURR')
    result.to_csv('only_nonan_logit1.csv')
# RandomForest
flag = 0
if flag == 0:
    pass
else:
    start = time.time()
    model = RandomForestClassifier(class_weight = 'balanced')
    #model.fit(X_train, y_train)
    #pred_train = model.predict(X_train)
    #pred_val = model.predict(X_val)
    model.fit(X,Y)
    end  = time.time()
    #print('F1 score of training set:',f1_score(pred_train, y_train, average='weighted'))
    #print('F1 score of test set:',f1_score(pred_val, y_val, average='weighted'))
    print('Done! Time spent:',end-start)
    pred_test = pd.DataFrame(model.predict_proba(test_X)).loc[:,1]
    result = pd.concat([test_ID,pred_test],axis=1)
    result.columns = ['SK_ID_CURR','TARGET']
    result = result.set_index('SK_ID_CURR')
    result.to_csv('only_nonan_rf1.csv')


# Gradient Boosting
flag = 0
if flag == 0:
    pass
else:
    start = time.time()
    model = XGBClassifier()
    #model.fit(X_train, y_train)
    #pred_train = model.predict(X_train)
    #pred_val = model.predict(X_val)
    model.fit(X,Y)
    end  = time.time()
    #print('F1 score of training set:',f1_score(pred_train, y_train, average='weighted'))
    #print('F1 score of test set:',f1_score(pred_val, y_val, average='weighted'))
    print('Done! Time spent:',end-start)
    pred_test = pd.DataFrame(model.predict_proba(test_X)).loc[:,1]
    result = pd.concat([test_ID,pred_test],axis=1)
    result.columns = ['SK_ID_CURR','TARGET']
    result = result.set_index('SK_ID_CURR')
    result.to_csv('only_nonan_xgb1.csv')


# Neural Network
'''
import tensorflow as tf

# Hyperparam
LR = 0.000001 # learning rate
ITERATION = 10000
BATCH_SIZE = 1500
KEEP_PROB = 0.7
NUM_FEAT = 135
NUM_CLASS = 2

class DataIter():
    def __init__(self, X,Y):
        self.X = X
        self.Y = Y
        self.size = len(self.X)
        self.epochs = 0
        self.df = pd.concat([X,Y],axis=1)
        self.pos = self.df.loc[self.Y == 1]
        self.neg = self.df.loc[self.Y == 0]
    def next_batch(self,n):
        #X_train,X_val,y_train,y_val = train_test_split(X,Y,test_size = n/self.size,random_state=seed,stratify=Y)
        #res = pd.concat([X_val,y_val],axis=1)
        pos_sample = self.pos.sample(n, replace=True)
        neg_sample = self.neg.sample(n, replace=True)
        res = pd.concat([neg_sample, pos_sample],axis=0)
        return res

# build graph
tf.reset_default_graph()
x = tf.placeholder(tf.float32,[BATCH_SIZE*2, NUM_FEAT])
y = tf.placeholder(tf.int32,[BATCH_SIZE*2])
keep_prob = tf.constant(KEEP_PROB)
nn_inputs = tf.layers.dense(x, units = round(0.75*NUM_FEAT), kernel_initializer = tf.truncated_normal_initializer(),activation = tf.nn.sigmoid)# hidden layer 1
nn_inputs = tf.nn.dropout(nn_inputs, KEEP_PROB)
#print(nn_inputs.get_shape)
nn_inputs = tf.layers.dense(x, units = round(0.5*NUM_FEAT), kernel_initializer = tf.truncated_normal_initializer(),activation = tf.nn.sigmoid) # hidden layer 2
nn_inputs = tf.nn.dropout(nn_inputs, KEEP_PROB)
#print(nn_inputs.get_shape)
nn_inputs = tf.layers.dense(x, units = round(0.25*NUM_FEAT), kernel_initializer = tf.truncated_normal_initializer(),activation = tf.nn.sigmoid) # hidden layer 3
nn_inputs = tf.nn.dropout(nn_inputs, KEEP_PROB)
#print(nn_inputs.get_shape)
nn_inputs = tf.layers.dense(x, units = 10, kernel_initializer = tf.truncated_normal_initializer(),activation = tf.nn.sigmoid) # hidden layer 4
nn_inputs = tf.nn.dropout(nn_inputs, KEEP_PROB)
with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [10, NUM_CLASS],initializer = tf.truncated_normal_initializer())
    b = tf.get_variable('b', [NUM_CLASS],initializer = tf.constant_initializer(0.0))
logits = tf.matmul(nn_inputs, W)+b
#print(logits.get_shape)
preds = tf.nn.softmax(logits)
prediction = tf.cast(tf.argmax(preds,1), tf.int32)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = preds))
precision, precision_op = tf.metrics.precision(y,prediction)
#print(precision.get_shape)
recall, recall_op = tf.metrics.recall(y,prediction)
#print(recall.get_shape)
f1score = 2*precision*recall/(precision+recall)
train_step = tf.train.AdamOptimizer(LR).minimize(loss)

# session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    tr = DataIter(X,Y)
    for i in range(ITERATION):
        batch = tr.next_batch(BATCH_SIZE)
        sess.run(train_step, feed_dict={x:batch.iloc[:,0:-1], y:batch['TARGET']})
        if i%1000 == 0:
            _,prec = sess.run([precision,precision_op], feed_dict={x:batch.iloc[:,0:-1], y:batch['TARGET']})
            _,rec = sess.run([recall,recall_op], feed_dict={x:batch.iloc[:,0:-1], y:batch['TARGET']})
            f1s = sess.run(f1score, feed_dict={x:batch.iloc[:,0:-1], y:batch['TARGET']})
            los = sess.run(loss, feed_dict={x:batch.iloc[:,0:-1], y:batch['TARGET']})
            print('losss after',i,'round',los)
            print('precision after',i,'round',prec)
            print('recall after',i,'round',rec)
            print('F1 score after',i,'round:',f1s)
            print('\n----------------------------------\n')
            print('logits:\n',sess.run(logits,feed_dict={x:batch.iloc[:,0:-1], y:batch['TARGET']}))
            print('preds:\n',sess.run(preds, feed_dict={x:batch.iloc[:,0:-1], y:batch['TARGET']}))
            print('prediction:\n',sess.run(prediction, feed_dict={x:batch.iloc[:,0:-1], y:batch['TARGET']}))
            print('y:\n',sess.run(y,feed_dict={x:batch.iloc[:,0:-1], y:batch['TARGET']}))
            print('\n----------------------------------\n')
    cursor = 0
    while cursor <= len(test_X):
        if cursor+BATCH_SIZE <= len(test_X):
            te = test_X.iloc[cursor:cursor+2*BATCH_SIZE]
        else:
            te = pd.concat([test_X.iloc[cursor:len(test_X)],test_X.iloc[0:2*BATCH_SIZE-len(test_X.iloc[cursor:len(test_X)])]])
        results = sess.run(preds, feed_dict={x:te})
        if cursor == 0:
            prediction = pd.DataFrame(data=results, columns=['0','TARGET'])
        else:
            prediction = pd.concat([prediction, pd.DataFrame(data=results,columns=['0','TARGET'])])
        cursor += 2*BATCH_SIZE
        '''
'''
result = prediction.iloc[0:len(test_X)]
pred_test = pd.DataFrame(result['TARGET']).reset_index()
result_final = pd.concat([test_ID, pred_test],axis=1).drop('index',axis=1)
result_final.columns = ['SK_ID_CURR','TARGET']
result_final = result_final.set_index('SK_ID_CURR')
result_final.to_csv('only_nonan_NN1.csv')
'''


