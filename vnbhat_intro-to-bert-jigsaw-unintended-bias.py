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

import sys

import json

import codecs

import numpy as np



sys.path.insert(0, '../input/pretrained-bert-including-scripts/master/bert-master')

!cp -r '../input/kerasbert/keras_bert' '/kaggle/working'

BERT_PRETRAINED_DIR = '../input/pretrained-bert-including-scripts/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12'

print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))



#Keras_bert packages

import keras

from keras_bert.keras_bert.bert import get_model

from keras_bert.keras_bert.loader import load_trained_model_from_checkpoint

from keras_bert.keras_bert import Tokenizer

from keras_bert.keras_bert import get_base_dict, get_model, gen_batch_inputs

from keras.callbacks import EarlyStopping

## To create and visualize a model

from tqdm import tqdm, tqdm_pandas

from tqdm import tqdm

from keras.models import Sequential

import tensorflow as tf

from keras.models import Model

from keras import backend as K

from keras.optimizers import Adam

from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda, Add, Flatten

from keras.callbacks import ModelCheckpoint
config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')

checkpoint_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')

max_len = 72

model = load_trained_model_from_checkpoint(config_file, checkpoint_file, training=True, seq_len=max_len)



model.summary(line_length=120)
bsz = 64 #batch-size

dict_path = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')

token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:

    for line in reader:

        token = line.strip()

        token_dict[token] = len(token_dict)

tokenizer = Tokenizer(token_dict)

train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv').sample(frac=0.4,random_state = 42)

train_labels = train['target']


def remove_Stopwords(pd_frame):

    from nltk.stem import WordNetLemmatizer 

    lemmatizer = WordNetLemmatizer()

    stop_words = {'not', 'won', 'but', 'this', 'most', "isn't", 'have', 'just', 'themselves', 'too', 'isn', 'hadn', 'about', 'from', 'both', "don't", 'hasn', "mightn't", 'your', 'his', 'than', 'so', 'now', 'until', 'they', 'ain', 'does', 'itself', 'or', 'off', "aren't", 'haven', 'i', 'you', 'he', 'why', 'it', 'under', 'd', 'mightn', 'up', 'each', 'down', 'y', 'o', 're', 'wouldn', "should've", 'no', 'which', 'aren', 'a', "you'll", "mustn't", 'doing', "didn't", 'same', 't', 'whom', "shan't", 'an', 'don', "wasn't", 'its', 'those', 'own', 'yours', 'myself', 'and', 'has', 'wasn', 'll', "hasn't", 'was', 'in', 'few', 'other', "couldn't", 'then', 'be', 'being', 'nor', "needn't", 'can', "won't", 'couldn', 'weren', 'been', 'for', "shouldn't", 'there', 'needn', 'yourself', 'how', 'her', 'herself', 'below', "you're", 'when', 'very', "haven't", 'into', 'didn', 'them', 'to', 'above', 'shan', 'some', 'are', 'on', 'is', 'their', 'at', 'am', 'hers', 'doesn', 'between', 'while', 's', 'should', 'theirs', 'himself', "that'll", 'ours', 'yourselves', 'what', 'again', 'had', 'ma', 'our', "you'd", 'my', 'out', "she's", 'she', 'if', "weren't", 'that', 'these', 'will', 'with', 'against', 'do', 'ourselves', 'all', 'who', 'as', 'here', "wouldn't", 've', 'through', 'the', 'after', "hadn't", 'of', 'having', 'once', 'only', 'because', 'where', "it's", 'by', 'shouldn', "doesn't", 'we', 'during', 'over', 'any', "you've", 'him', 'me', 'more', 'did', 'further', 'such', 'm', 'mustn', 'before', 'were'}

    main_np = np.empty([len(pd_frame), 72])

    for i in tqdm(range(len(pd_frame))):

        new_str = ''

        tot_length = 70

        sent = pd_frame.iloc[i]

        for j in sent.split():

            if j not in stop_words:

                new_str += ' ' + lemmatizer.lemmatize(j.lower())

        tokens = tokenizer.encode(new_str, max_len = 72)[0]

        for k in range(72):

            main_np[i,k] = tokens[k]

    return main_np


def auc(y_true, y_pred):

    auc = tf.metrics.auc(y_true, y_pred)[1]

    K.get_session().run(tf.local_variables_initializer())

    return auc



adam = Adam(lr=2e-5,decay=0.01)

maxlen = 72



#checkpointer = ModelCheckpoint(filepath='best_model.hdf5', verbose=5, save_best_only=True)



es = EarlyStopping(monitor=auc, mode='min', verbose=1)

#start_model = Sequential()

sequence_output  = model.layers[-6].output

pool_output = Dense(1, activation='sigmoid',kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),name = 'real_output')(sequence_output)



print('begin_build')

full_model = Model(inputs=model.input, outputs=pool_output)

full_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[auc])

print('begin training')

full_model.summary()
token_input = remove_Stopwords(train['comment_text'])

seg_input = np.zeros((token_input.shape[0],maxlen))

mask_input = np.ones((token_input.shape[0],maxlen))

full_model.fit([token_input, seg_input, mask_input], train_labels, validation_split=0.3, epochs = 1, 

               callbacks=[es])
token_input2 = remove_Stopwords(pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')['comment_text'])

seg_input2 = np.zeros((token_input2.shape[0],maxlen))

mask_input2 = np.ones((token_input2.shape[0],maxlen))

pred = full_model.predict([token_input2, seg_input2, mask_input2])
keras.utils.plot_model(full_model, show_shapes=True)
test_info = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

new_df = {'id':test_info['id'], 'prediction' : pred}

def extract_pred(pred):

    ans = []

    for i in pred:

        ans.append(i[0])

    return ans

df = pd.DataFrame({"id": test_info["id"], "prediction": extract_pred(pred)})

df.to_csv("submission.csv", index=False)