# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json
import tensorflow as tf

print ("Read Dataset ... ")
train = json.load(open("../input/train.json"))
test = json.load(open("../input/test.json"))
print ("Prepare text data of Train and Test ... ")
train_text = [" ".join(doc["ingredients"]).lower() for doc in train]
test_text = [" ".join(doc["ingredients"]).lower() for doc in test]
target = [doc["cuisine"] for doc in train]

print ("{} different cuisines".format(len(set(target))))

# Feature Engineering 
print ("TF-IDF on text data ... ")
tfidf = TfidfVectorizer(binary=True)
X = tfidf.fit_transform(train_text)
X_test = tfidf.transform(test_text)

X = X.toarray()
X_test = X_test.toarray()

# Label Encoding - Target 
print ("Label Encode the Target Variable ... ")
lb = LabelEncoder()
y = lb.fit_transform(target)


train_X,test_X, train_y, test_y = train_test_split(X, y, test_size = 0.1, random_state = 0)
print ("{} training samples and {} validation samples".format(train_X.shape[0], test_X.shape[0]))

class NN():
  def __init__(self, feature_size, num_classes, learning_rate=0.001):
    self.feature_size = feature_size
    self.num_classes = num_classes
    self.learning_rate = learning_rate

    with tf.name_scope("placeholder"):
      self.input = tf.placeholder(tf.float32, [None, feature_size], name='input')
      self.target = tf.placeholder(tf.int32, [None], name='target')
      self.dropout = tf.placeholder(tf.float32, name = 'dropout')
    with tf.name_scope("inference"):
      hidden_units = 512
      W0 = tf.Variable(tf.truncated_normal([feature_size, hidden_units]))
      b0 = tf.Variable(tf.zeros(hidden_units))
      h = tf.matmul(self.input, W0)+ b0
      h = tf.nn.relu(h)
      h = tf.nn.dropout(h, keep_prob = self.dropout)
      W = tf.Variable(tf.truncated_normal([hidden_units, num_classes]))
      b = tf.Variable(tf.zeros(num_classes))

      self.logits = tf.matmul(h, W)+b
      
    with tf.name_scope("loss"):
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.logits, labels = self.target)
      self.loss =tf.reduce_sum(losses)

    with tf.name_scope('optimize'):
      self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
    with tf.name_scope('prediction'):
      self.prob = tf.nn.softmax(self.logits)
      self.prediction = tf.argmax(self.prob, axis = 1)
      corrects = tf.equal(tf.cast(self.target, tf.int64), self.prediction)
      self.accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

if __name__ == '__main__':
  print ("Train the model......")
  model = NN(feature_size = train_X.shape[1], num_classes = len(set(target)))
  num_epoch  = 80
  batch_size = 128
  with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for epoch in range(num_epoch):


      num_batches = int((train_X.shape[0]-1)/batch_size)+1
      for batch in range(num_batches):
        start_index = batch_size*batch
        end_index = min(train_X.shape[0], (batch+1)*batch_size)
        feed_train = {model.input: train_X[start_index:end_index],
                      model.target: train_y[start_index:end_index],
                      model.dropout: 0.5}
        _, loss, accuracy = sess.run([model.train_op, model.loss, model.accuracy], feed_dict = feed_train)
        #print("\tepoch {}, \tbatch {}, \tloss {:g}, \tacc {:g}".format(epoch, batch, loss, accuracy))

      feed_val = {
                 model.input: test_X,
                 model.target: test_y,
                 model.dropout: 1.0
      }
      loss_val, accuracy_val = sess.run([model.loss, model.accuracy], feed_dict = feed_val)
      print ("\tepoch {}, \tloss {}, \tacc {:g}".format(epoch, loss_val, accuracy_val))

    print ("predict on test data......")
    feed_test = {model.input: X_test,
                 model.dropout: 1.0}
    predictions = sess.run(model.prediction, feed_dict=feed_test)
    prediction_labels = lb.inverse_transform(predictions)
    test_ids = [doc['id'] for doc in test]
    submission = pd.DataFrame({'id':test_ids, 'cuisine':prediction_labels}, columns = ['id', 'cuisine'])
    submission.to_csv("submission.csv", index = False)