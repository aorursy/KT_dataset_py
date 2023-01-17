!pip install git+https://www.github.com/keras-team/keras-contrib.git


import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow_hub as hub

from keras.models import Model

import tensorflow as tf

import re

from keras.engine.topology import Layer

from sklearn.model_selection import train_test_split

import keras.backend as K

from keras.layers import Dense , Input, PReLU , Dropout

from keras_contrib.callbacks import CyclicLR



import os

print(os.listdir("../input"))

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



train_data['title'] = train_data['title'].apply(lambda s : re.sub(r'[^\w\s]','',s))

test_data['title'] = test_data['title'].apply(lambda s : re.sub(r'[^\w\s]','',s))



train_text = train_data['title'].tolist()

train_text = [' '.join(t.split()) for t in train_text]

train_text = np.array(train_text, dtype=object)[:, np.newaxis]



train_labels = train_data['Category'].values



n_classes = len(train_data['Category'].unique())



X_train , X_val, y_train  , y_val = train_test_split(train_text , 

                                                     train_labels , 

                                                     stratify = train_labels , 

                                                     train_size = 0.8,

                                                     random_state = 100)



test_text = test_data['title'].tolist()

test_text = [' '.join(t.split()) for t in test_text]

test_text = np.array(test_text, dtype=object)[:, np.newaxis]



#Reference https://towardsdatascience.com/elmo-embeddings-in-keras-with-tensorflow-hub-7eb6f0145440

class ElmoEmbeddingLayer(Layer):

    def __init__(self, trainable = True ,**kwargs):

        self.dimensions = 1024

        self.trainable = trainable

        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable, name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))

        super(ElmoEmbeddingLayer, self).build(input_shape)

        

    def call(self, x, mask=None):

        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),

                          as_dict=True,

                          signature='default',

                          )['default']

        return result

    

    def compute_mask(self, inputs, mask=None):

        return K.not_equal(inputs, '--PAD--')

    

    def compute_output_shape(self, input_shape):

        return (input_shape[0], self.dimensions)
def gen_model(n_classes = 2) :

    inp = Input(shape = (1,) , name = 'input' , dtype = tf.string)

    embedding = ElmoEmbeddingLayer()(inp)

    dense = Dense(512 )(embedding)

    dense = PReLU()(dense)

    dense = Dropout(0.3)(dense)

    dense = Dense(256)(dense)

    dense = PReLU()(dense)

    dense = Dropout(0.3)(dense)

    if n_classes > 1 :

        pred = Dense(n_classes, activation='softmax')(dense)

    else :

        pred = Dense(n_classes, activation='sigmoid')(dense)

    model = Model(inputs=inp, outputs=pred)

    model.compile(loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'] , optimizer = 'nadam')

    return model
K.clear_session()

model = gen_model(n_classes)

#model = gen_model(58)

model.summary()



batch_size = 512

epochs = 20

step_size = int(int(len(X_train)/batch_size)*epochs/2)



cb = [

    CyclicLR(5e-4 , 2e-3 , step_size)

]
model.fit(X_train, 

          y_train,

          validation_data = (X_val, y_val),

          epochs = epochs,

          batch_size= batch_size,

          verbose = 1,

          shuffle = True,

          callbacks = cb)
preds = model.predict(test_text,verbose = 1).argmax(axis = 1)
sub_df = test_data[['itemid']].copy()

sub_df['Category'] = preds

sub_df.to_csv('submission.csv' , index = False)

sub_df.head()