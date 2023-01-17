
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import tensorflow as tf 
import numpy as np 
import pandas as pd
import random

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#load data from file
word_embeddings = np.load("../input/preprocessed-data/word_embeddings.npy")
word_id_train_val = np.load("../input/preprocessed-data/word_id_train.npy")
sequence_len_train_val = np.load("../input/preprocessed-data/sequence_len_train.npy")
sentiment_vectors_train_val = np.load("../input/preprocessed-data/sentiment_vectors_train.npy")

word_id_test = np.load("../input/preprocessed-data/word_id_test.npy")
sequence_len_test = np.load("../input/preprocessed-data/sequence_len_test.npy")

seq_len = len(word_id_train_val[0])
VOCAB_LENGTH,EMBEDDING_DIMENSION = word_embeddings.shape

rnn_size = EMBEDDING_DIMENSION
learning_rate = 0.01
learning_rate_dacay = 0.9
num_layers = 2
num_labels = 5

with tf.name_scope("inputs"):
  inputs = tf.placeholder(tf.int32, [None,seq_len])
  targets = tf.placeholder(tf.float32, [None,num_labels])
  input_sequence_length = tf.placeholder(tf.int32, [None,])
  BATCH_SIZE = tf.placeholder(tf.int32, [], name='BATCH_SIZE')

with tf.name_scope("embedding"):
  glove_weights_initializer = tf.constant_initializer(word_embeddings)
  embedding_weights = tf.get_variable(name='embedding_weights', shape=(VOCAB_LENGTH, EMBEDDING_DIMENSION), initializer=glove_weights_initializer,trainable=True)
  embedded_inputs = tf.nn.embedding_lookup(embedding_weights, inputs)

#RNN cells
with tf.name_scope("rnn"):
  cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
  cell =  tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = 0.5)
  cell = tf.nn.rnn_cell.MultiRNNCell([cell for _ in range(num_layers)])
  
  outputs, state = tf.nn.dynamic_rnn(cell= cell, inputs = embedded_inputs, sequence_length = input_sequence_length, dtype = tf.float32)
  #state dimension: [num_layers, 2, batch_size, rnn_size]
  last = state[-1][1]

#output layer
with tf.name_scope("score"):
  weights = tf.Variable(tf.random_normal([rnn_size, num_labels]))
  bias = tf.Variable(tf.random_normal([num_labels]))
  logits = tf.nn.bias_add(tf.matmul(last, weights), bias)
  #logits = tf.layers.dense(last, num_labels)

# calculate loss and create optimizer
with tf.name_scope("optimize"):
  loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits)
  total_loss = tf.reduce_mean(loss)
  trainable_vars = tf.trainable_variables()
  grads,_ = tf.clip_by_global_norm(tf.gradients(total_loss, trainable_vars), 5.0)
  lr = tf.Variable(0.0, trainable=False)
  optimizer = tf.train.AdamOptimizer(lr)
  train_op = optimizer.apply_gradients(zip(grads, trainable_vars))
  #train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

with tf.name_scope("accuracy"):
  predictions = tf.argmax(logits, axis = 1) #按行查找
  correct_pred = tf.equal(predictions, tf.argmax(targets,1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#split the data into training set and validation set 8:2
length_training = int (len(word_id_train_val)*0.8)
word_id_train = word_id_train_val[:length_training]
word_id_val = word_id_train_val[length_training:]
sequence_len_train = sequence_len_train_val[:length_training]
sequence_len_val = sequence_len_train_val[length_training:]
sentiment_vectors_train = sentiment_vectors_train_val[:length_training]
sentiment_vectors_val = sentiment_vectors_train_val[length_training:]

print ("the number of samples in training set: ", len(word_id_train))
print ("the number of samples in validation set: ", len(word_id_val) )


def shuffle_data(word_id, sentiment, sequence_len):
  ids = list(range(len(word_id)))
  random.shuffle(ids) #shuffle will change the list in-place
  return [word_id[i] for i in ids], [sentiment[i] for i in ids], [sequence_len[i] for i in ids]

def get_batches(word_id, sentiment, sequence_len, batch_size):
  data_batches, labels_batches, length_batches = [], [], []
  for i in range(len(word_id)//batch_size):
    data_batch = word_id[i*batch_size: (i+1)*batch_size]
    labels_batch = sentiment[i*batch_size: (i+1)*batch_size]
    length_batch = sequence_len[i*batch_size: (i+1)*batch_size]
    data_batches.append(data_batch)
    labels_batches.append(labels_batch)
    length_batches.append(length_batch)
  return data_batches, labels_batches, length_batches

#feed data into NN and start training
num_epoches = 4
batch_size = 128
num_batches = len(word_id_train)//batch_size
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:

  sess.run(init_op)
  for epoch in range(num_epoches):
    sess.run(tf.assign(lr, learning_rate *
                               (learning_rate_dacay ** (epoch))))
    training_accuracy = []
    training_loss = []
    shuffled_word_id_train, shuffled_sentiment_train, shuffled_sequence_len_train = shuffle_data(word_id_train, sentiment_vectors_train, sequence_len_train)
    data_batches, labels_batches, length_batches = get_batches(shuffled_word_id_train, shuffled_sentiment_train, shuffled_sequence_len_train, batch_size)
    
    for batch in range(num_batches):
      feed = {inputs: data_batches[batch], targets: labels_batches[batch], input_sequence_length: length_batches[batch], BATCH_SIZE:batch_size}
      loss,_, train_accuracy = sess.run([total_loss, train_op, accuracy], feed_dict=feed)
      training_loss.append(loss)
      training_accuracy.append(train_accuracy)
      if (batch%100 == 0):
        data_batches_val, labels_batches_val, length_batches_val = get_batches(word_id_val, sentiment_vectors_val,sequence_len_val, batch_size)
        val_accuracy = []
        for i in range(len(word_id_val)//batch_size):
          feed_val = {inputs: data_batches_val[i], targets: labels_batches_val[i], input_sequence_length: length_batches_val[i], BATCH_SIZE:batch_size}
          accuracy_i = sess.run(accuracy, feed_dict=feed_val)
          val_accuracy.append(accuracy_i)
        
       
        print('Epoch {:>3}/{} - Training Loss: {:>6.3f} - Training accuracy: {:>6.3f} - Validation accuracy: {:>6.3f}'
             .format( epoch+1,
                      num_epoches,  
                      loss,
                      train_accuracy,
                      np.mean(val_accuracy)
                      ))

      
    print('Epoch {:>3}/{} - Epoch Training Loss: {:>6.3f} - Epoch Training accuracy: {:>6.3f}'
             .format( epoch+1,
                      num_epoches,  
                      np.mean(training_loss),
                      np.mean(training_accuracy)
                      ))

  

  #make predictions on test data
  predictions = []

  for i in range(len(word_id_test)):

    feed_prediction = {inputs: [word_id_test[i]], 
                      input_sequence_length: [sequence_len_test[i]],
                      BATCH_SIZE:1 }

 
    logit = sess.run(logits, feed_dict=feed_prediction)
    predictions.append(np.argmax(logit, axis = 1)[0])

  data = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep='\t')
  data['Sentiment'] = predictions
  data.drop(['Phrase', 'SentenceId'], axis=1, inplace=True)
  print(data.head(10))
  data.to_csv('Submission.csv', header=True, index=None, sep=',')

