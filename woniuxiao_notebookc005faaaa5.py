# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

train_features.head()
import tensorflow as tf

import numpy as np

train_features["cp_type"]=train_features["cp_type"].replace(["trt_cp","ctl_vehicle"],[1,0])

train_features["cp_dose"]=train_features["cp_dose"].replace(["D1","D2"],[1,0])

train_features["cp_time"]=train_features["cp_time"].replace([24,48,72],[0,1,2])



train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

x= train_features.drop(['sig_id'],axis=1)

y= train_targets_scored.drop(['sig_id'],axis=1)



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)



x_train=x_train.values.tolist()

x_train = np.array(x_train)



x_test=x_test.values.tolist()

x_test = np.array(x_test)



y_train=y_train.values.tolist()

y_train = np.array(y_train)



y_test=y_test.values.tolist()

y_test = np.array(y_test)



x_train
train_targets_scored.head()
x_train.shape[0]
class MLP(tf.keras.Model):

    def __init__(self):

        super().__init__()

        self.flatten = tf.keras.layers.Flatten()    # Flatten层将除第一维（batch_size）以外的维度展平

        self.dense1 = tf.keras.layers.Dense(units=800, activation=tf.nn.relu) 

        self.dense2 = tf.keras.layers.Dense(units=600, activation=tf.nn.relu)

        self.dense3 = tf.keras.layers.Dense(units=500, activation=tf.nn.relu)

        self.dense4 = tf.keras.layers.Dense(units=206)



    def call(self, inputs):         

        x = self.flatten(inputs)   

        x = self.dense1(x)          

        x = self.dense2(x)

        x = self.dense3(x)

        x = self.dense4(x)

        output = tf.nn.softmax(x)

        return output

    

def get_batch(batch_size):

        # 从数据集中随机取出batch_size个元素并返回

        batch_size=int(batch_size)

        index = np.random.randint(0, x_train.shape[0], batch_size)

        return x_train[index,:], y_train[index,:]

    

num_epochs = 5

batch_size = 50

learning_rate = 0.00005



model = MLP()

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)



num_batches = int(x_train.shape[0] // batch_size * num_epochs)

for batch_index in range(num_batches):

    index = np.random.randint(0, x_train.shape[0], batch_size)

    X=x_train[index,:]

    y=y_train[index,:]

#     y = y.tolist()

#     y = tf.constant(y)

    with tf.GradientTape() as tape:

        y_pred = model(X)

#         y_pred = np.array(y_pred)

#         y_pred = y_pred.reshape(y_pred.shape[0]*y_pred.shape[1],1)

#         y_pred = y_pred.tolist()

#         y_pred = tf.constant(y_pred)

        

        loss = tf.keras.losses.binary_crossentropy(y_true=y, y_pred=y_pred)

        loss = tf.reduce_mean(loss)

        print("batch %d: loss %f" % (batch_index, loss.numpy()))

    grads = tape.gradient(loss, model.variables)

    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

num_batches = int(x_test.shape[0] // batch_size)

for batch_index in range(num_batches):

    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size

    y_predd = model.predict(x_test[start_index: end_index])

    y_predd = np.array(y_predd)

    y_predd = y_predd.reshape(y_predd.shape[0]*y_predd.shape[1],1)

    

    y_truee=y_test[start_index: end_index]

    y_truee = np.array(y_truee)

    y_truee = y_truee.reshape(y_predd.shape[0]*y_predd.shape[1],1)

    sparse_categorical_accuracy.update_state(y_true=y_truee, y_pred=y_predd)

print("test accuracy: %f" % sparse_categorical_accuracy.result())
test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

test_features.head()
test_features["cp_type"]=test_features["cp_type"].replace(["trt_cp","ctl_vehicle"],[1,0])

test_features["cp_dose"]=test_features["cp_dose"].replace(["D1","D2"],[1,0])

test_features["cp_time"]=test_features["cp_time"].replace([24,48,72],[0,1,2])

test_featuress = test_features.drop(['sig_id'],axis=1)

test_featuress_predict = model.predict(test_featuress)

test_featuress_predict
sample_submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

sample_submission.head()
sample_submission1 = sample_submission.drop(['sig_id'],axis=1)

columnn =sample_submission1.columns

columnn
res = pd.DataFrame(test_featuress_predict,columns=columnn)

pd.set_option('display.float_format',lambda x : '%.19f' % x)

res.insert(0,'sig_id',sample_submission["sig_id"])

res.head()
res.to_csv('/kaggle/working/submission.csv', index=0, encoding = "utf-8")
def score(real,pred):

    eps=1e-15

    pred = np.clip(pred, eps, 1 - eps)

    ret = 0

    rett = 0

    m = len(real[0])

    

    n = len(real)

    print(m,n)

    for j in range(m):

        for i in range(n):

            ret = ret + (real[i][j]*np.log(pred[i][j])+(1-real[i][j])*np.log(1-pred[i][j]))

        rett = rett + (ret / n)

    return rett / -m

score(y_test,model.predict(x_test))
from sklearn.metrics import log_loss

log_loss(y_test,model.predict(x_test))