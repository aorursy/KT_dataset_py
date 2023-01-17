import logging

import numpy as np

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



logging.basicConfig(level=logging.DEBUG, format='%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s')

logger = logging.getLogger(__name__)
import pandas as pd



train_data = pd.read_table('../input/quora-question-pairs/train.csv', sep=',')[['question1', 'question2', 'is_duplicate']].dropna().values

test_data = pd.read_table('../input/quora-question-pairs/test.csv', sep=',')[['question1', 'question2']].fillna('').values

train_data_context, train_data_label = train_data[:, 0:2], train_data[:, 2]

print(train_data_context.shape)
from sklearn.model_selection import train_test_split



train_split_data_context, valid_split_data_context, train_split_data_label, valid_split_data_label = train_test_split(train_data_context, train_data_label, test_size=0.05)



train_split_data_context = [line[0] + ' ' + line[1] for line in train_split_data_context]

valid_split_data_context = [line[0] + ' ' + line[1] for line in valid_split_data_context]



print(train_split_data_context[0])

print(valid_split_data_context[0])
from tqdm import tqdm

from tensorflow.keras.preprocessing.text import Tokenizer



tokenizer = Tokenizer(

    num_words=50000, # 词表去20000，词表的提取根据TF的计算结果排序

    filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ',

    lower=True,

    split=' ',

    char_level=False,

    oov_token=None,

    document_count=0)



tokenizer.fit_on_texts(tqdm(train_split_data_context))
from keras.preprocessing.sequence import pad_sequences





train_split_data_index = np.array(tokenizer.texts_to_sequences(tqdm(train_split_data_context)))

valid_split_data_index = np.array(tokenizer.texts_to_sequences(tqdm(valid_split_data_context)))

train_split_data_index = pad_sequences(tqdm(train_split_data_index), maxlen=50, padding='post')

valid_split_data_index = pad_sequences(tqdm(valid_split_data_index), maxlen=50, padding='post')



print(train_split_data_index[0])

print(valid_split_data_index[0])
import random



def get_batch(epoches, batch_size, data, label):

    data = list(zip(train_split_data_index, label))

    for epoch in range(epoches):

        random.shuffle(data)

        for batch in range(0, len(data), batch_size):

            if batch + batch_size >= len(data):

                yield data[batch: len(data)]

            else:

                yield data[batch: (batch + batch_size)]
import tensorflow as tf



class MyModel(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dims):

        super(MyModel, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dims)

        self.logiticRegression_one = tf.keras.layers.Dense(1, input_shape=(embedding_dims,), activation="sigmoid")

        self.logiticRegression_two = tf.keras.layers.Dense(2, activation='softmax')



    def call(self, sentence):

        embedding = self.embedding(sentence)

        sentence_embedding = tf.reduce_mean(embedding, axis=1)

        result = self.logiticRegression_two(sentence_embedding)

        print("for_return: ", result)

        return result
model = MyModel(vocab_size=50005,

                embedding_dims=128)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)



test_loss = tf.keras.metrics.Mean(name='test_loss')

train_loss = tf.keras.metrics.Mean(name='train_loss')

test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')

train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')



@tf.function

def test_step(sentence, label):

    predictions = model(sentence)

    labels = tf.concat((tf.expand_dims(1 - label, axis=-1), tf.expand_dims(label, axis=-1)), axis=1)

    losses = tf.nn.weighted_cross_entropy_with_logits(labels, predictions, pos_weight=0.3632292393, name=None)

    

    predict_label = tf.argmax(predictions, axis=1)

    correct_predict = tf.equal(tf.cast(label, dtype=tf.int32), tf.cast(predict_label, dtype=tf.int32))



    test_loss(losses)

    test_accuracy(correct_predict)





@tf.function

def train_step(sentence, label):

    with tf.GradientTape() as tape:

        predictions = model(sentence)

        labels = tf.concat((tf.expand_dims(1 - label, axis=-1), tf.expand_dims(label, axis=-1)), axis=1)

        losses = tf.nn.weighted_cross_entropy_with_logits(labels, predictions, pos_weight=0.3632292393, name=None)

        

        predict_label = tf.argmax(predictions, axis=1)

        correct_predict = tf.equal(tf.cast(label, dtype=tf.int32), tf.cast(predict_label, dtype=tf.int32))



    gradients = tape.gradient(losses, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(losses)

    train_accuracy(correct_predict)
EPOCHS, BATCH_SIZE = 10, 320

train_data_index = tf.convert_to_tensor(train_split_data_index, dtype=tf.int32)

train_data_label = tf.convert_to_tensor(train_split_data_label, dtype=tf.float32)

valid_data_index = tf.convert_to_tensor(valid_split_data_index, dtype=tf.int32)

valid_data_label = tf.convert_to_tensor(valid_split_data_label, dtype=tf.float32)

train_ds = tf.data.Dataset.from_tensor_slices((train_data_index, train_data_label)).repeat(EPOCHS).batch(BATCH_SIZE)



step = 0

for data, label in train_ds:

    step += 1

    train_step(data, label)

    if step % 100 == 0:

        test_step(valid_data_index, valid_data_label)

        template = 'Step {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'

        print(template.format(step,

                              train_loss.result().numpy(),

                              train_accuracy.result().numpy(),

                              test_loss.result().numpy(),

                              test_accuracy.result().numpy()))