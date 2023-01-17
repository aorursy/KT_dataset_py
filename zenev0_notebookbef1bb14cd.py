# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



sess = tf.InteractiveSession()



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv', header=0)



data["Died"] = 0

data["Died"][data["Survived"] == 0] = 1



data.Age = data.Age.fillna(data.Age.median())



data['IsMale'] = data['Sex'].map( {'female': 0, 'male': 1, -1:-1} ).astype(int)



#data.Embarked = data.Embarked.fillna(-1)

#data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q':2, -1:-1} ).astype(int)



data = pd.concat([data, pd.get_dummies(data['Embarked'], prefix='Embarked')], axis=1)

print(data.info())

data = data.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'Cabin'], axis=1)



percent_train = .7

num_train = int(len(data)*percent_train)

train = data[:num_train]

test = data[num_train:]



train_x = train.drop(['Survived', 'Died'], axis=1)

train_y = train[[ 'Survived', 'Died']]



test_x = test.drop(['Survived', 'Died'], axis=1)

test_y = test[[ 'Survived', 'Died']]
train.head(20)

#print(train.Age.unique())

#print(train.head(4))

#train.info()
x = tf.placeholder(tf.float32, shape=[None, 10])

y_ = tf.placeholder(tf.float32, shape=[None, 2])



W = tf.Variable(tf.truncated_normal([10,10], stddev=0.1))

b = tf.Variable(tf.constant(.1, shape=[10]))



W2 = tf.Variable(tf.truncated_normal([10,6], stddev=0.1))

b2 = tf.Variable(tf.constant(.1, shape=[6]))



W3 = tf.Variable(tf.truncated_normal([6,2], stddev=0.1))

b3 = tf.Variable(tf.constant(.1, shape=[2]))



layer1= tf.matmul(x,W) + b

layer1_relu = tf.nn.relu(layer1)

layer2 = tf.matmul(layer1_relu,W2) + b2

layer2_relu = tf.nn.relu(layer2)

y = tf.matmul(layer2_relu, W3) + b3



cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))



train_step = tf.train.AdamOptimizer().minimize(cross_entropy)



correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))



accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



sess.run(tf.initialize_all_variables())
bucket_size = 10000

start = 0

print_num = 1000

for i in range(10000):

    end = start + bucket_size

    if end >= len(train_y):

        end = len(train_y)

    

    train_step.run(feed_dict={x: train_x[start:end], y_: train_y[start:end]})

    if i % print_num == 0:

        print("entropy", cross_entropy.eval(feed_dict={x: train_x[start:end], y_: train_y[start:end]}))

        print("train", accuracy.eval(feed_dict={x: train_x[start:end], y_: train_y[start:end]}))

        print("test", accuracy.eval(feed_dict={x: test_x, y_: test_y}))

    start = end

    if end >= len(train_y):

        start = 0

    

print("test", accuracy.eval(feed_dict={x: test_x, y_: test_y}))
print("W", W.eval())

print("b", b.eval())

print("W2", W2.eval())

print("b2", b2.eval())

print(test.Survived.mean())