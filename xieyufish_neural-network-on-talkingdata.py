# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

#import seaborn as sns

import matplotlib.pyplot as plt

import os

from sklearn.preprocessing import LabelEncoder

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import StratifiedKFold

from sklearn.metrics import log_loss



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
datadir = '../input'

# 把device_id作为行号，后面避免了搜索device的程序

# 注意这里的ga_train和ga_test要一起处理，因为app_events等文件没有区分train和test，因此索引都是全部一起索引

ga_train = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'), index_col='device_id')

ga_test = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'), index_col='device_id') 

# 将行指标设置为event_id，这样就可以通过app_events找到app对应的event_id，然后通过event_id指标直接索引到device_id，最后通过ga_train将device_id的指标直接索引到gender和age

events = pd.read_csv(os.path.join(datadir,'events.csv'), index_col='event_id', parse_dates=['timestamp']) 

# 所有的app的is_installed都为1，因此此列无效，将其删去

app_events = pd.read_csv(os.path.join(datadir,'app_events.csv'), usecols=['event_id','app_id','is_active'])

# phone_brand里面有重复（对应同一个大品牌下面的不同device_model）

##### 此处需要分析一下重复的数据

device_brand = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))

device_brand = device_brand.drop_duplicates('device_id').set_index('device_id')

app_labels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))
ga_train['trainrow'] = np.arange(ga_train.shape[0])

ga_test['testrow'] = np.arange(ga_test.shape[0])
# 将brand编码成整数，可使用transform和inverse_transform进行brand和编码的转换

brand_encoder = LabelEncoder().fit(device_brand['phone_brand'])

device_brand['brand'] = brand_encoder.transform(device_brand['phone_brand'])

ga_train['brand'] = device_brand['brand'] # 由于行标为device_id，因此device_id自动匹配

ga_test['brand'] = device_brand['brand']

# 建立一个稀疏矩阵，行是device（对应的trainrow）,列是各个brand，值为1代表某个device对应是某个brand

Xtr_brand = csr_matrix((np.ones(ga_train.shape[0]), (ga_train['trainrow'], ga_train['brand'])))

Xte_brand = csr_matrix((np.ones(ga_test.shape[0]), (ga_test['testrow'], ga_test['brand'])))

print(Xtr_brand.shape, Xte_brand.shape)
m = device_brand.phone_brand.str.cat(device_brand.device_model)

modelencoder = LabelEncoder().fit(m)

device_brand['model'] = modelencoder.transform(m)

ga_train['model'] = device_brand['model']

ga_test['model'] = device_brand['model']

Xtr_model = csr_matrix((np.ones(ga_train.shape[0]), 

                       (ga_train.trainrow, ga_train.model)))

Xte_model = csr_matrix((np.ones(ga_test.shape[0]), 

                       (ga_test.testrow, ga_test.model)))

print('Model features: train shape {}, test shape {}'.format(Xtr_model.shape, Xte_model.shape))
# 将所有的app编码

app_encoder = LabelEncoder().fit(app_events['app_id'])

app_events['app'] = app_encoder.transform(app_events['app_id'])
# 将app_events与events合并

device_apps = app_events.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)

# 将相同的device合并，并记录每个device使用app的次数

device_apps = device_apps.groupby(['device_id','app'])['app'].agg(['size'])

# 将device_apps继续与ga_train和ga_test合并（仅合并行标），从而可以通过trainrow和testrow得到它们对应的分类

device_apps = device_apps.merge(ga_train[['trainrow']], how='left', left_index=True, right_index=True)

device_apps = device_apps.merge(ga_test[['testrow']], how='left', left_index=True, right_index=True)

device_apps = device_apps.reset_index() # 原来是将device_id和app都设为行标，现在将其恢复为属性
napps = len(app_encoder.classes_)

# 建立一个稀疏矩阵，行是device（对应的trainrow/testrow）,列是各个app，值为1代表某个device对应安装了某个app

d = device_apps.dropna(subset=['trainrow']) # 取出有trainrow（testrow为NaN）的数据

Xtr_app = csr_matrix((np.ones(d.shape[0]), (d['trainrow'], d['app'])), shape=[ga_train.shape[0],napps])

d = device_apps.dropna(subset=['testrow']) 

Xte_app = csr_matrix((np.ones(d.shape[0]), (d['testrow'], d['app'])), shape=[ga_test.shape[0],napps])

# 对应有app信息的设备数量大于有品牌信息的设备数量，说明不是所有的device都有对应的brand

print(Xtr_app.shape, Xte_app.shape)
# 将app编号加入到app_labels中

# 因为app_labels里面有一些app是在events中没有出现的，因此只取出那些出现了的

app_labels = app_labels.loc[app_labels.app_id.isin(app_events.app_id.unique())]

app_labels['app'] = app_encoder.transform(app_labels['app_id'])

# 将label重新编号

label_encoder = LabelEncoder().fit(app_labels['label_id'])

app_labels['label'] = label_encoder.transform(app_labels['label_id'])
device_labels = (device_apps[['device_id','app']]

                .merge(app_labels[['app','label']])

                .groupby(['device_id','label'])['app'].agg(['size'])

                .merge(ga_train[['trainrow']], how='left', left_index=True, right_index=True)

                .merge(ga_test[['testrow']], how='left', left_index=True, right_index=True)

                .reset_index())
nlabels = len(label_encoder.classes_) # 下面csr_matrix后面要加一个shape，不然可能由于中间函数筛选的原因使得大小不一致

d = device_labels.dropna(subset=['trainrow'])

Xtr_label = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.label)), shape=(ga_train.shape[0],nlabels))

d = device_labels.dropna(subset=['testrow'])

Xte_label = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.label)), shape=(ga_test.shape[0],nlabels))

print(Xtr_label.shape, Xte_label.shape)
X = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label), format='csr')

X_test = hstack((Xte_brand, Xte_model, Xte_app, Xte_label), format='csr')

print(X.shape, X_test.shape)
from sklearn.preprocessing import LabelBinarizer

target_encoder = LabelEncoder().fit(ga_train['group'])

target_encoder1 = LabelBinarizer(sparse_output=True).fit(ga_train['group'])

#Y1 = target_encoder1.transform(ga_train['group'])

#target_encoder2 = LabelBinarizer().fit(Y1)

Y = target_encoder1.transform(ga_train['group'])

nclasses = len(target_encoder1.classes_)

#app_labels

print(nclasses)
Xtest = X[70001:]

ytest = Y[70001:]



Xtrain = X[:70001]

ytrain = Y[:70001]
import tensorflow as tf

def weight_variable(shape):

    initial = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(initial)



def bias_variable(shape):

    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial)



N = 100

M = 30

# Create the model

x = tf.placeholder(tf.float32, [None, 21527])

W1 = weight_variable([21527, N])

b1 = bias_variable([N])

y1 = tf.nn.relu(tf.matmul(x, W1) + b1)

keep_prob1 = tf.placeholder("float")

fc1_drop = tf.nn.dropout(y1, keep_prob1)



W2 = weight_variable([N, M])

b2 = bias_variable([M])

y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)

keep_prob2 = tf.placeholder("float")

fc2_drop = tf.nn.dropout(y1, keep_prob2)



W3 = weight_variable([M, 12])

b3 = bias_variable([12])

y = tf.matmul(y2, W3) + b3



# Define loss and optimizer

y_ = tf.placeholder(tf.float32, [None, 12])



# The raw formulation of cross-entropy,

#

#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),

#                                 reduction_indices=[1]))

#

# can be numerically unstable.

#

# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw

# outputs of 'y', and then average across the batch.

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

#reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.0001), 

                                             #weights_list=[W1, W2])

loss = cross_entropy

#train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)



sess = tf.InteractiveSession()

tf.global_variables_initializer().run()
j0 = 30

i0 = 100

n0 = 100

# Train

for j in range(j0):

    for i in range(i0):

        shuffle = np.random.choice(70000, size=100, replace=False)

        sess.run(train_step, feed_dict={x: Xtrain[shuffle].toarray(), 

                                        y_: ytrain[shuffle].toarray(), 

                                        keep_prob1:0.5, 

                                        keep_prob2:0.5})

    if j%5==0:

        # Test trained model

        #shuffle_test = np.random.choice(74645, size=10000, replace=False)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #print('epoch:', j, sess.run(cross_entropy, feed_dict={x: X[shuffle_test].toarray(),

                                                              #y_: Y[shuffle_test].toarray()}))

        print('epoch:', j, sess.run(cross_entropy, feed_dict={x: Xtest.toarray(),

                                                              y_: ytest.toarray(), 

                                                              keep_prob1:1.0, 

                                                              keep_prob2:1.0}))
logloss = tf.nn.softmax(y)

i1 = 1121

n0 = 100



for i in range(i1):

    if n0*(i+1)>112071:

        m0 = 112071

    else:

        m0 = n0*(i+1)

    if i==0:

        pred_nn = sess.run(logloss, feed_dict={x: X_test[n0*i:m0].toarray()})

    else:

        pred_nn = np.row_stack((pred_nn, sess.run(logloss, feed_dict={x: X_test[n0*i:m0].toarray(), 

                                                                     keep_prob1:1.0, 

                                                                     keep_prob2:1.0})))



#pred_nn = pd.DataFrame(sess.run(log_loss, feed_dict={x: X_test.toarray()}), index=ga_test.index, columns=targetencoder.classes) 

#pred_nn.to_csv('logreg_subm_neuralnetwork.csv',index=True)
target_encoder = LabelEncoder().fit(ga_train['group'])

pred = pd.DataFrame(pred_nn, index=ga_test.index, columns=target_encoder1.classes_) 

pred.to_csv('testnew_nn.csv',index=True)

pred.head()