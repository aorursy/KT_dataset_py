# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import transformers

import tensorflow as tf

from scipy.stats import spearmanr

import tensorflow_hub as hub

import re

from sklearn.preprocessing import MinMaxScaler

import gc

from sklearn.model_selection import GroupKFold,KFold

from scipy.stats import spearmanr, rankdata

from sklearn.linear_model import MultiTaskElasticNet

import string

from collections import Counter

from gensim.models import Word2Vec

from transformers import XLNetConfig, TFXLNetModel, XLNetTokenizer, TFXLNetMainLayer, BertConfig, TFBertMainLayer, BertTokenizer, TFBertModel

import lightgbm as lgb
DIR = '/kaggle/input/google-quest-challenge'
xlnet_tokenizer = XLNetTokenizer.from_pretrained('/kaggle/input/xlnetbasecased/tokenizer')

bert_tokenizer = BertTokenizer.from_pretrained('/kaggle/input/bertbaseuncasedcomplete/bert-model')
train_df = pd.read_csv(DIR+'/train.csv')

test_df = pd.read_csv(DIR+'/test.csv')

cols = train_df.columns[11:]
def tokenize_input(tokenizer, s1, s2, tags, data_name, max_length, tokenizer_name = 'bert'):

  if s2 is not None:

    x = tokenizer.encode_plus(s1, s2, pad_to_max_length=False)

    if len(x['input_ids']) > max_length:

      segment_1 = int(0.25*max_length)

      x['input_ids'] = x['input_ids'][:segment_1] + x['input_ids'][-(max_length-segment_1):]

      x['attention_mask'] = x['attention_mask'][:segment_1] + x['attention_mask'][-(max_length-segment_1):]

      x['token_type_ids'] = x['token_type_ids'][:segment_1] + x['token_type_ids'][-(max_length-segment_1):]

    else:

      diff = max_length - len(x['input_ids'])

      if tokenizer_name == 'xlnet':

        x['input_ids'] = [tokenizer.pad_token_id]*diff + x['input_ids']

        x['attention_mask'] = [0]*diff + x['attention_mask']

        x['token_type_ids'] = [tokenizer.pad_token_type_id]*diff + x['token_type_ids']

      else:

        x['input_ids'] = x['input_ids'] + [tokenizer.pad_token_id]*diff

        x['attention_mask'] = x['attention_mask'] + [0]*diff

        x['token_type_ids'] = x['token_type_ids'] + [0]*diff

      

  else:

    x = tokenizer.encode_plus(s1)

    if len(x['input_ids']) > max_length:

      segment_1 = int(0.25*max_length)

      x['input_ids'] = x['input_ids'][:segment_1] + x['input_ids'][-(max_length-segment_1):]

      x['attention_mask'] = x['attention_mask'][:segment_1] + x['attention_mask'][-(max_length-segment_1):]

      x['token_type_ids'] = x['token_type_ids'][:segment_1] + x['token_type_ids'][-(max_length-segment_1):]

    else:

      diff = max_length - len(x['input_ids'])

      if tokenizer_name == 'xlnet':

        x['input_ids'] = [tokenizer.pad_token_id]*diff + x['input_ids']

        x['attention_mask'] = [0]*diff + x['attention_mask']

        x['token_type_ids'] = [tokenizer.pad_token_type_id]*diff + x['token_type_ids']

      else:

        x['input_ids'] = x['input_ids'] + [tokenizer.pad_token_id]*diff

        x['attention_mask'] = x['attention_mask'] + [0]*diff

        x['token_type_ids'] = x['token_type_ids'] + [tokenizer.pad_token_type_id]*diff

  

  data[data_name][tags[0]].append(x['input_ids']) 

  data[data_name][tags[1]].append(x['token_type_ids'])

  data[data_name][tags[2]].append(x['attention_mask']) 



data = {}

# ******************************************XLNET*************************************************************************************

data['xlnet_train_t_a'] = {}

data['xlnet_train_q_a'] = {}

data['xlnet_train_t_q'] = {}

data['xlnet_train_q'] = {}

data['xlnet_train_a'] = {}



data['xlnet_test_t_a'] = {}

data['xlnet_test_q_a'] = {}

data['xlnet_test_t_q'] = {}

data['xlnet_test_q'] = {}

data['xlnet_test_a'] = {}



tags = ['input_ids', 'token_type_ids', 'attention_masks']

data['xlnet_train_t_a'][tags[0]], data['xlnet_train_t_a'][tags[1]], data['xlnet_train_t_a'][tags[2]] = [], [], []

data['xlnet_train_q_a'][tags[0]], data['xlnet_train_q_a'][tags[1]], data['xlnet_train_q_a'][tags[2]] = [], [], []

data['xlnet_train_t_q'][tags[0]], data['xlnet_train_t_q'][tags[1]], data['xlnet_train_t_q'][tags[2]] = [], [], []

data['xlnet_train_q'][tags[0]], data['xlnet_train_q'][tags[1]], data['xlnet_train_q'][tags[2]] = [], [], []

data['xlnet_train_a'][tags[0]], data['xlnet_train_a'][tags[1]], data['xlnet_train_a'][tags[2]] = [], [], []





data['xlnet_test_t_a'][tags[0]], data['xlnet_test_t_a'][tags[1]], data['xlnet_test_t_a'][tags[2]] = [], [], []

data['xlnet_test_q_a'][tags[0]], data['xlnet_test_q_a'][tags[1]], data['xlnet_test_q_a'][tags[2]] = [], [], []

data['xlnet_test_t_q'][tags[0]], data['xlnet_test_t_q'][tags[1]], data['xlnet_test_t_q'][tags[2]] = [], [], []

data['xlnet_test_q'][tags[0]], data['xlnet_test_q'][tags[1]], data['xlnet_test_q'][tags[2]] = [], [], []

data['xlnet_test_a'][tags[0]], data['xlnet_test_a'][tags[1]], data['xlnet_test_a'][tags[2]] = [], [], []





for i in range(train_df.shape[0]):

  tokenize_input(xlnet_tokenizer, train_df.loc[i, 'question_title'], train_df.loc[i, 'answer'], tags, 'xlnet_train_t_a', 512, 'xlnet')

  tokenize_input(xlnet_tokenizer, train_df.loc[i, 'question_body'], train_df.loc[i, 'answer'], tags, 'xlnet_train_q_a', 512, 'xlnet')

  tokenize_input(xlnet_tokenizer, train_df.loc[i, 'question_title'], train_df.loc[i, 'question_body'], tags, 'xlnet_train_t_q', 512, 'xlnet')

  tokenize_input(xlnet_tokenizer, train_df.loc[i, 'question_body'], None, tags, 'xlnet_train_q', 512, 'xlnet')

  tokenize_input(xlnet_tokenizer, train_df.loc[i, 'answer'], None, tags, 'xlnet_train_a', 512, 'xlnet')

for i in range(test_df.shape[0]):

  tokenize_input(xlnet_tokenizer, test_df.loc[i, 'question_title'], test_df.loc[i, 'answer'], tags, 'xlnet_test_t_a', 512, 'xlnet')

  tokenize_input(xlnet_tokenizer, test_df.loc[i, 'question_body'], test_df.loc[i, 'answer'], tags, 'xlnet_test_q_a', 512, 'xlnet')

  tokenize_input(xlnet_tokenizer, test_df.loc[i, 'question_title'], test_df.loc[i, 'question_body'], tags, 'xlnet_test_t_q', 512, 'xlnet')

  tokenize_input(xlnet_tokenizer, test_df.loc[i, 'question_body'], None, tags, 'xlnet_test_q', 512, 'xlnet')

  tokenize_input(xlnet_tokenizer, test_df.loc[i, 'answer'], None, tags, 'xlnet_test_a', 512, 'xlnet')



# ******************************************BERT*************************************************************************************



data['bert_train_t_a'] = {}

data['bert_train_q_a'] = {}

data['bert_train_t_q'] = {}

data['bert_train_q'] = {}

data['bert_train_a'] = {}



data['bert_test_t_a'] = {}

data['bert_test_q_a'] = {}

data['bert_test_t_q'] = {}

data['bert_test_q'] = {}

data['bert_test_a'] = {}



data['bert_train_t_a'][tags[0]], data['bert_train_t_a'][tags[1]], data['bert_train_t_a'][tags[2]] = [], [], []

data['bert_train_q_a'][tags[0]], data['bert_train_q_a'][tags[1]], data['bert_train_q_a'][tags[2]] = [], [], []

data['bert_train_t_q'][tags[0]], data['bert_train_t_q'][tags[1]], data['bert_train_t_q'][tags[2]] = [], [], []

data['bert_train_q'][tags[0]], data['bert_train_q'][tags[1]], data['bert_train_q'][tags[2]] = [], [], []

data['bert_train_a'][tags[0]], data['bert_train_a'][tags[1]], data['bert_train_a'][tags[2]] = [], [], []



data['bert_test_t_a'][tags[0]], data['bert_test_t_a'][tags[1]], data['bert_test_t_a'][tags[2]] = [], [], []

data['bert_test_q_a'][tags[0]], data['bert_test_q_a'][tags[1]], data['bert_test_q_a'][tags[2]] = [], [], []

data['bert_test_t_q'][tags[0]], data['bert_test_t_q'][tags[1]], data['bert_test_t_q'][tags[2]] = [], [], []

data['bert_test_q'][tags[0]], data['bert_test_q'][tags[1]], data['bert_test_q'][tags[2]] = [], [], []

data['bert_test_a'][tags[0]], data['bert_test_a'][tags[1]], data['bert_test_a'][tags[2]] = [], [], []



for i in range(train_df.shape[0]):

  tokenize_input(bert_tokenizer, train_df.loc[i, 'question_title'], train_df.loc[i, 'answer'], tags, 'bert_train_t_a', 512, 'bert')

  tokenize_input(bert_tokenizer, train_df.loc[i, 'question_body'], train_df.loc[i, 'answer'], tags, 'bert_train_q_a', 512, 'bert')

  tokenize_input(bert_tokenizer, train_df.loc[i, 'question_title'], train_df.loc[i, 'question_body'], tags, 'bert_train_t_q', 512, 'bert')

  tokenize_input(bert_tokenizer, train_df.loc[i, 'question_body'], None, tags, 'bert_train_q', 512, 'bert')

  tokenize_input(bert_tokenizer, train_df.loc[i, 'answer'], None, tags, 'bert_train_a', 512, 'bert')

for i in range(test_df.shape[0]):

  tokenize_input(bert_tokenizer, test_df.loc[i, 'question_title'], test_df.loc[i, 'answer'], tags, 'bert_test_t_a', 512, 'bert')

  tokenize_input(bert_tokenizer, test_df.loc[i, 'question_body'], test_df.loc[i, 'answer'], tags, 'bert_test_q_a', 512, 'bert')

  tokenize_input(bert_tokenizer, test_df.loc[i, 'question_title'], test_df.loc[i, 'question_body'], tags, 'bert_test_t_q', 512, 'bert')

  tokenize_input(bert_tokenizer, test_df.loc[i, 'question_body'], None, tags, 'bert_test_q', 512, 'bert')

  tokenize_input(bert_tokenizer, test_df.loc[i, 'answer'], None, tags, 'bert_test_a', 512, 'bert')



for key, _ in data.items():

  for k, _ in data[key].items():

    data[key][k] = np.array(data[key][k])
def SpearmanCorrCoeff(A, B):

  overall_score = 0

  for index in range(A.shape[1]):

      overall_score += spearmanr(A[:, index], B[:, index]).correlation

  return overall_score/30

class PredictCallback(tf.keras.callbacks.Callback):

  def __init__(self, data, labels):

    self.data = data

    self.labels = labels

  def on_epoch_end(self, epoch, logs = {}):

    predictions = self.model.predict(self.data)

    print('\n\t Validation Score - ' + str(SpearmanCorrCoeff(self.labels, predictions)))
class BERT(TFBertModel):

  def __init__(self, config, *inputs, **kwrgs):

    super(BERT, self).__init__(config, *inputs, **kwrgs)

    self.bert = TFBertMainLayer(config, name = 'bert')

    for i in range(1, 45):

      self.bert.submodules[-i].trainable = False

      

  def call(self, inputs, **kwrgs):

    outputs = self.bert(inputs)

    hidden_states = outputs[2]

    h12 = hidden_states[-1][:, 0, :]

    h11 = hidden_states[-2][:, 0, :]

    h10 = hidden_states[-3][:, 0, :]

    h9 = hidden_states[-4][:, 0, :]

    concat = tf.keras.layers.Concatenate(axis = -1)([h9, h10, h11, h12])

    return concat



class XLNet(TFXLNetModel):

  def __init__(self, config, *inputs, **kwrgs):

    super(XLNet, self).__init__(config, *inputs, **kwrgs)

    self.transformer = TFXLNetMainLayer(config, name = 'transformer')

    for i in range(1, 3):

      self.transformer.layer[-i].trainable = False

  def call(self, inputs, **kwrgs):

    outputs = self.transformer(inputs)

    hidden_states = outputs[1]

    h12 = hidden_states[-1][:, 0, :]

    h11 = hidden_states[-2][:, 0, :]

    h10 = hidden_states[-3][:, 0, :]

    h9 = hidden_states[-4][:, 0, :]

    concat = tf.keras.layers.Concatenate(axis = -1)([h9, h10, h11, h12])

    return concat
def create_model(name):

  id_1 = tf.keras.Input(shape = (512), dtype = tf.int32)

  id_2 = tf.keras.Input(shape = (512), dtype = tf.int32)



  type_id_1 = tf.keras.Input(shape = (512), dtype = tf.int32)

  type_id_2 = tf.keras.Input(shape = (512), dtype = tf.int32)

    

  a1 = tf.keras.Input(shape = (512), dtype = tf.int32)

  a2 = tf.keras.Input(shape = (512), dtype = tf.int32)

  if name == 'xlnet':

    config = XLNetConfig.from_pretrained('/kaggle/input/xlnetbasecased/config-xlnet-base-cased', output_hidden_states = True)

    transformer = XLNet.from_pretrained('/kaggle/input/xlnetbasecased/model-xlnet-base-cased', config = config)

                                                

  else:

    config = BertConfig.from_pretrained('/kaggle/input/bertbaseuncasedcomplete/bert-model', output_hidden_states = True)

    transformer = BERT.from_pretrained('/kaggle/input/bertbaseuncasedcomplete/bert-model', config = config)

  

  out_1 = transformer({'input_ids':id_1, 'attention_mask':a1, 'token_type_ids':type_id_1})

  out_2 = transformer({'input_ids':id_2, 'attention_mask':a2, 'token_type_ids':type_id_2})

  

  concat = tf.keras.layers.Concatenate(axis = -1)([out_1, out_2])

  drop = tf.keras.layers.Dropout(rate = 0.1)(concat)

  dense = tf.keras.layers.Dense(30, activation = 'sigmoid')(drop)

  return tf.keras.Model(inputs = [id_1, id_2, type_id_1, type_id_2, a1, a2], outputs = [dense])
main_fold = np.random.randint(0, 5)

print(main_fold)
gkf = GroupKFold(n_splits = 5).split(X = train_df.url, groups = train_df.url)

for fold, (train_idx, valid_idx) in enumerate(gkf):

  if fold != main_fold:

    continue

  final_outputs = train_df[cols].values.astype(np.float16)

  tf.keras.backend.clear_session()

  xlnet_train_inputs = (

                    data['xlnet_train_a'][tags[0]][train_idx], data['xlnet_train_t_q'][tags[0]][train_idx], data['xlnet_train_a'][tags[1]][train_idx], data['xlnet_train_t_q'][tags[1]][train_idx],

                   data['xlnet_train_a'][tags[2]][train_idx], data['xlnet_train_t_q'][tags[2]][train_idx]  

                  

                 )

  xlnet_valid_inputs = (

                    data['xlnet_train_a'][tags[0]][valid_idx], data['xlnet_train_t_q'][tags[0]][valid_idx], data['xlnet_train_a'][tags[1]][valid_idx], data['xlnet_train_t_q'][tags[1]][valid_idx],

                   data['xlnet_train_a'][tags[2]][valid_idx], data['xlnet_train_t_q'][tags[2]][valid_idx]  

                  

                  )

#   bert_train_inputs = (

#                     data['bert_train_a'][tags[0]][train_idx], data['bert_train_t_q'][tags[0]][train_idx], data['bert_train_a'][tags[1]][train_idx], data['bert_train_t_q'][tags[1]][train_idx],

#                    data['bert_train_a'][tags[2]][train_idx], data['bert_train_t_q'][tags[2]][train_idx]  

                  

#                  )

#   bert_valid_inputs = (

#                     data['bert_train_a'][tags[0]][valid_idx], data['bert_train_t_q'][tags[0]][valid_idx], data['bert_train_a'][tags[1]][valid_idx], data['bert_train_t_q'][tags[1]][valid_idx],

#                    data['bert_train_a'][tags[2]][valid_idx], data['bert_train_t_q'][tags[2]][valid_idx]  

                  

#                   )



  

  xlnet_model = create_model('xlnet')

  xlnet_model.compile(tf.keras.optimizers.Adam(learning_rate = 2*1e-5), loss = tf.keras.losses.BinaryCrossentropy())

  xlnet_model.fit(x = xlnet_train_inputs, y = final_outputs[train_idx], epochs = 3, batch_size = 4, steps_per_epoch = train_idx.shape[0]//4)

  tf.keras.backend.clear_session()

  print("\n####################################################################################################################\n")

#   bert_model = create_model('bert')

#   bert_model.compile(tf.keras.optimizers.Adam(learning_rate = 2.3*1e-5), loss = tf.keras.losses.BinaryCrossentropy())

#   bert_model.fit(x = bert_train_inputs, y = final_outputs[train_idx], epochs = 2, batch_size = 4, steps_per_epoch = train_idx.shape[0]//4)



  break
xlnet_inputs = (

                    data['xlnet_train_a'][tags[0]], data['xlnet_train_t_q'][tags[0]], data['xlnet_train_a'][tags[1]], data['xlnet_train_t_q'][tags[1]],

                   data['xlnet_train_a'][tags[2]], data['xlnet_train_t_q'][tags[2]]  

                  

                 )

predictions = xlnet_model.predict(xlnet_inputs)

train_y = final_outputs[train_idx]

valid_y = final_outputs[valid_idx]

train_preds = predictions[train_idx]

valid_preds = predictions[valid_idx]
estimators = []

post_valid_preds = np.zeros_like(valid_preds)

for i in range(train_preds.shape[1]):

  lgb_train = lgb.Dataset(train_preds[:, i:i+1], label=train_y[:, i])

  params = {'objective': 'rmse', 'num_leaves': 3, 'learning_rate': 0.01, 'min_data_in_leaf': 20}

  est = lgb.train(params, lgb_train, num_boost_round=50)

  estimators.append(est)

  post_valid_preds[:, i] = est.predict(valid_preds[:, i:i+1])

  if (post_valid_preds[:, i].max() - post_valid_preds[:, i].min()) < 0.000001:

    max_idx = np.argmax(post_valid_preds[:, i])

    post_valid_preds[:, i][max_idx] = min(0.9999999, post_valid_preds[:, i][max_idx] + 0.001)

    min_idx = np.argmin(post_valid_preds[:, i])

    post_valid_preds[:, i][min_idx] = max(0.0000001, post_valid_preds[:, i][min_idx] - 0.001)

    

print(f"Validation Score (Before) {SpearmanCorrCoeff(valid_y, valid_preds)}")

print(f"Validation Score (After) {SpearmanCorrCoeff(valid_y, post_valid_preds)}")
xlnet_test_inputs = inputs = (data['xlnet_test_a'][tags[0]], data['xlnet_test_t_q'][tags[0]], data['xlnet_test_a'][tags[1]], 

    data['xlnet_test_t_q'][tags[1]],data['xlnet_test_a'][tags[2]], data['xlnet_test_t_q'][tags[2]])

test_preds = xlnet_model.predict(xlnet_test_inputs)

post_test_preds = np.zeros_like(test_preds)



for i in range(train_preds.shape[1]):

    post_test_preds[:, i] = estimators[i].predict(test_preds[:, i:i+1])

    if (post_valid_preds[:, i].max() - post_valid_preds[:, i].min()) < 0.000001:

        max_idx = np.argmax(post_valid_preds[:, i])

        post_valid_preds[:, i][max_idx] = min(0.9999999, post_valid_preds[:, i][max_idx] + 0.00001)

        min_idx = np.argmin(post_valid_preds[:, i])

        post_valid_preds[:, i][min_idx] = max(0.0000001, post_valid_preds[:, i][min_idx] - 0.00001)

submission = pd.read_csv(DIR+'/sample_submission.csv')

submission.iloc[:,1:] = post_test_preds

submission.to_csv("submission.csv", index = False)

submission.head()