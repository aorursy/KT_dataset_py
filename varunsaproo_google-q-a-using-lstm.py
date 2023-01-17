import pandas as pd

import numpy as np

from gensim.models import Word2Vec, KeyedVectors

import string

import re

from collections import Counter

import gensim.downloader as api

from nltk.stem import PorterStemmer, SnowballStemmer

from nltk.stem.lancaster import LancasterStemmer

import tensorflow as tf

import tensorflow_hub as hub

from sklearn.model_selection import GroupKFold

from scipy.stats import spearmanr

import os

import gc

import warnings

warnings.filterwarnings("ignore")
vector_dim = 300

encoder = hub.load("/kaggle/input/universalsentenceencoderqa")
ps = PorterStemmer()

lc = LancasterStemmer()

sb = SnowballStemmer('english')
contractions = {

"ain't": "is not",

"aren't": "are not",

"can't": "cannot",

"can't've": "cannot have",

"'cause": "because",

"could've": "could have",

"couldn't": "could not",

"couldn't've": "could not have",

"didn't": "did not",

"doesn't": "does not",

"don't": "do not",

"hadn't": "had not",

"hadn't've": "had not have",

"hasn't": "has not",

"haven't": "have not",

"he'd": "he would",

"he'd've": "he would have",

"he'll": "he will",

"he'll've": "he he will have",

"he's": "he is",

"how'd": "how did",

"how'd'y": "how do you",

"how'll": "how will",

"how's": "how is",

"I'd": "I would",

"i'd've": "I would have",

"i'll": "I will",

"i'll've": "I will have",

"i'm": "I am",

"i've": "I have",

"i'd": "i would",

"i'd've": "i would have",

"i'll": "i will",

"i'll've": "i will have",

"i'm": "i am",

"i've": "i have",

"isn't": "is not",

"it'd": "it would",

"it'd've": "it would have",

"it'll": "it will",

"it'll've": "it will have",

"it's": "it is",

"let's": "let us",

"ma'am": "madam",

"mayn't": "may not",

"might've": "might have",

"mightn't": "might not",

"mightn't've": "might not have",

"must've": "must have",

"mustn't": "must not",

"mustn't've": "must not have",

"needn't": "need not",

"needn't've": "need not have",

"o'clock": "of the clock",

"oughtn't": "ought not",

"oughtn't've": "ought not have",

"shan't": "shall not",

"sha'n't": "shall not",

"shan't've": "shall not have",

"she'd": "she would",

"she'd've": "she would have",

"she'll": "she will",

"she'll've": "she will have",

"she's": "she is",

"should've": "should have",

"shouldn't": "should not",

"shouldn't've": "should not have",

"so've": "so have",

"so's": "so as",

"that'd": "that would",

"that'd've": "that would have",

"that's": "that is",

"there'd": "there would",

"there'd've": "there would have",

"there's": "there is",

"they'd": "they would",

"they'd've": "they would have",

"they'll": "they will",

"they'll've": "they will have",

"they're": "they are",

"they've": "they have",

"to've": "to have",

"wasn't": "was not",

"we'd": "we would",

"we'd've": "we would have",

"we'll": "we will",

"we'll've": "we will have",

"we're": "we are",

"we've": "we have",

"weren't": "were not",

"what'll": "what will",

"what'll've": "what will have",

"what're": "what are",

"what's": "what is",

"what've": "what have",

"when's": "when is",

"when've": "when have",

"where'd": "where did",

"where's": "where is",

"where've": "where have",

"who'll": "who will",

"who'll've": "who will have",

"who's": "who is",

"who've": "who have",

"why's": "why is",

"why've": "why have",

"will've": "will have",

"won't": "will not",

"won't've": "will not have",

"would've": "would have",

"wouldn't": "would not",

"wouldn't've": "would not have",

"y'all": "you all",

"y'all'd": "you all would",

"y'all'd've": "you all would have",

"y'all're": "you all are",

"y'all've": "you all have",

"you'd": "you would",

"you'd've": "you would have",

"you'll": "you will",

"you'll've": "you will have",

"you're": "you are",

"you've": "you have"

}



rules = {

    "'t": " not",

    "'cause": " because",

    "'ve": " have",

    "'t": " not",

    "'s": " is",

    "'d": " had"

}



punctuations = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']',

          '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^',

          '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',

          '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶',

          '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼',

          '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',

          'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',

          '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√'] + list(string.punctuation)



spell_replace = {

    'usepackage':'latex',

    'orf19':'gene',

    'documentclass':'latex',

    'magento':'open-source e-commerce',

    'appium':'web-app',

    'tikz':'programming language',

    'tikzpicture':'programming language',

    'openvpn':'vpn',

    'httpclient':'http client',

    'arraylist':'array list',

    'jsonobject': 'json',

    'artifactid':'xml',

    'hwnd':'os'

    

}

special_chars = ",  .  \"  :  )  (  -  !  ?  |  ;  '  $  &  /  [  ]  >  %  =  #  *  +  \  •  ~  @  £  ·  {  }  ©  ^  ®  <  →  °  €  ™  ›  ♥  ←  ×  §  ″  ′  Â  █  ½  à  …  “  ★  ”  –  ●  â  ►  −  ¢  ²  ¬  ░  ¶  ↑  ±  ¿  ═  ¦  ║  ―  ¥  ▓  —  ‹  ─  ▒  ：  ¼  ⊕  ▼  ▪  †  ■  ’  ▀  ¨  ▄  ♫  ☆  é  ¯  ♦  ¤  ▲  è  ¸  ¾  Ã  ⋅  ‘  ∞  ∙  ）  ↓  、  │  （  »  ，  ♪  ╩  ╚  ³  ・  ╦  ╣  ╔  ╗  ▬  ❤  ï  Ø  ¹  ≤  ‡  √  !  \"  $  %  &  '  (  )  *  +  ,  -  .  /  :  ;  <  =  >  ?  @  [  \  ]  ^  {  |  }  ~ ".split()
train_df = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')

test_df = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')

cols = train_df.columns[11:]

train_df['combined_t_q'] = train_df[['question_title', 'question_body']].apply(lambda x: x['question_title']+' '+x['question_body'], axis = 1)

test_df['combined_t_q'] = test_df[['question_title', 'question_body']].apply(lambda x: x['question_title']+' '+x['question_body'], axis = 1)

train_df['combined_t_a'] = train_df[['question_title', 'answer']].apply(lambda x: x['question_title']+' '+x['answer'], axis = 1)

test_df['combined_t_a'] = test_df[['question_title', 'answer']].apply(lambda x: x['question_title']+' '+x['answer'], axis = 1)
final_outputs = train_df[cols].values

unique_outputs = [np.sort(train_df[col].unique())[np.newaxis, :] for col in cols]
def preprocess_text(s):

    s = s.lower()

    #Expand Contractions

    for key, value in contractions.items():

        s = s.replace(key, f' {value} ')

    for key, value in rules.items():

        s = s.replace(key, f' {value} ')

    for punct in punctuations:

        if punct in special_chars:

            s = s.replace(punct, f' {punct} ')

        else:

            s = s.replace(punct, ' ')

    for key, value in spell_replace.items():

        s = s.replace(key, value)

    s = re.sub('\s+', ' ', s)

    return s
train_df['clean_answer'] = train_df['answer'].apply(preprocess_text)

train_df['clean_t_q'] = train_df['combined_t_q'].apply(preprocess_text)



test_df['clean_answer'] = test_df['answer'].apply(preprocess_text)

test_df['clean_t_q'] = test_df['combined_t_q'].apply(preprocess_text)
all_texts = train_df.clean_answer.tolist() + train_df.clean_t_q.tolist() + test_df.clean_answer.tolist() + test_df.clean_t_q.tolist() 

all_texts = [text.split() for text in all_texts]

counter = Counter()

for text in all_texts:

  counter.update(text)
vocab = {}

vocab['token2id'] = {key:id+1 for id, (key, _) in enumerate(counter.items())}

vocab['id2token'] = {value:key for key, value in vocab['token2id'].items()}

vocab['word_freq'] = dict(counter)
def build_embedding_matrix(vocab, texts, embedd_size, model):

  n = len(vocab['token2id'])+1

  embedding_matrix = np.zeros((n, embedd_size))

  for text in texts:

    for key in text:

      word = key

      try:

        embedding_matrix[vocab['token2id'][word]] = model.wv[word]

        continue

      except:

        pass

  return embedding_matrix
model = Word2Vec(size=300, window = 5, min_count = 1)

model.build_vocab(all_texts)

total_examples = model.corpus_count

model.intersect_word2vec_format('/kaggle/input/fasttext/fasttext-wiki-news-subwords-300', lockf=1.0)

model.train(all_texts, total_examples=total_examples, epochs=5)

embedding_matrix = build_embedding_matrix(vocab, all_texts, 300, model)
def get_token_ids(texts, max_length):

  tokens = []

  for text in texts:

    tmp_tokens = []

    if len(text.split()) > max_length:

      for each in (text.split()[:(max_length//2)] + text.split()[-(max_length//2):]):

        tmp_tokens.append(vocab['token2id'][each])

      tokens.append(tmp_tokens)

    else:

      for each in (text.split()[:max_length]):

        tmp_tokens.append(vocab['token2id'][each])

      tokens.append(tmp_tokens)

  return tf.keras.preprocessing.sequence.pad_sequences(tokens, padding="post", maxlen=max_length)
MAX_LENGTH = 250

data = {}

data['train_question_title'] = get_token_ids(train_df['clean_t_q'], MAX_LENGTH)

data['train_answer'] = get_token_ids(train_df['clean_answer'], MAX_LENGTH)

data['test_question_title'] = get_token_ids(test_df['clean_t_q'], MAX_LENGTH)

data['test_answer'] = get_token_ids(test_df['clean_answer'], MAX_LENGTH)
data['train_question_title_use'] = []

data['train_answer_use'] = []



data['test_question_title_use'] = []

data['test_answer_use'] = []



BATCH_SIZE = 4



for i in range(0, train_df.shape[0], BATCH_SIZE):

  data['train_question_title_use'] += [encoder.signatures['question_encoder'](tf.constant(train_df['clean_t_q'].iloc[i:i+BATCH_SIZE].tolist()))['outputs'].numpy().astype(np.float16)]

  data['train_answer_use'] += [encoder.signatures['response_encoder'](input = tf.constant(train_df['clean_answer'].iloc[i:i+BATCH_SIZE].tolist()), 

                                                                context = tf.constant(train_df['clean_answer'].iloc[i:i+BATCH_SIZE].tolist()))['outputs'].numpy().astype(np.float16)]

    

for i in range(0, test_df.shape[0], BATCH_SIZE):

  data['test_question_title_use'] += [encoder.signatures['question_encoder'](tf.constant(test_df['clean_t_q'].iloc[i:i+BATCH_SIZE].tolist()))['outputs'].numpy().astype(np.float16)]

  data['test_answer_use'] += [encoder.signatures['response_encoder'](input = tf.constant(test_df['clean_answer'].iloc[i:i+BATCH_SIZE].tolist()), 

                                                                context = tf.constant(test_df['clean_answer'].iloc[i:i+BATCH_SIZE].tolist()))['outputs'].numpy().astype(np.float16)]

    



data['train_question_title_use'] = np.vstack(data['train_question_title_use'])

data['train_answer_use'] = np.vstack(data['train_answer_use'])



data['test_question_title_use'] = np.vstack(data['test_question_title_use'])

data['test_answer_use'] = np.vstack(data['test_answer_use'])
def get_generator(X, batch_size = 32, training = True):

    Y = X[1]

    N = X[0][0].shape[0]

    if training == True:

        indexes = np.random.permutation(N)

    else:

        indexes = np.arange(N)

    def generator():

        for i in indexes:

            yield {"input_1": X[0][0][i], "input_2": X[0][1][i],"input_3": X[0][2][i], "input_4": X[0][3][i]}, Y[i]

    return tf.data.Dataset.from_generator(generator, 

    output_types = ({"input_1": tf.int32, "input_2": tf.int32,"input_3": tf.float16,"input_4": tf.float16}, tf.float16)).repeat().batch(batch_size)



def SpearmanCorrCoeff(A, B):

  overall_score = 0

  for index in range(A.shape[1]):

      overall_score += spearmanr(A[:, index], B[:, index]).correlation

  return np.round(overall_score/A.shape[1], 3)



class PredictCallback(tf.keras.callbacks.Callback):

  def __init__(self, data, labels):

    self.data = data

    self.labels = labels

  def on_epoch_end(self, epoch, logs = {}):

    predictions = self.model.predict(self.data)

    print('\nValidation Score - ' + str(SpearmanCorrCoeff(self.labels, predictions)))

    





class EWA(tf.keras.callbacks.Callback):

  def on_train_batch_end(self, batch, logs = None):

    global prev_weights

    if prev_weights is None:

      prev_weights = [tf.identity(x) for x in self.model.trainable_variables]

    else:

      beta = 0.1

      for index, _ in enumerate(self.model.trainable_variables):

        self.model.trainable_variables[index].assign(self.model.trainable_variables[index]*beta + (1-beta)*prev_weights[index])

        prev_weights = [tf.identity(x) for x in self.model.trainable_variables]
def create_model():

  i1 = tf.keras.Input(shape = (MAX_LENGTH), dtype = tf.int32)

  i2 = tf.keras.Input(shape = (MAX_LENGTH), dtype = tf.int32)

  i3 = tf.keras.Input(shape = (512), dtype = tf.float16)

  i4 = tf.keras.Input(shape = (512), dtype = tf.float16)



  e1 = tf.keras.layers.Embedding(input_dim = embedding_matrix.shape[0], output_dim = embedding_matrix.shape[1], weights = [embedding_matrix], trainable = False)(i1)

  e2 = tf.keras.layers.Embedding(input_dim = embedding_matrix.shape[0], output_dim = embedding_matrix.shape[1], weights = [embedding_matrix], trainable = False)(i2)

  

  sd_i1 = tf.keras.layers.SpatialDropout1D(0.2)(e1)

  sd_i2 = tf.keras.layers.SpatialDropout1D(0.2)(e2)



  lstm_q_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences = True))(sd_i1)

  lstm_q_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences = True))(lstm_q_1)

    

  lstm_a_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences = True))(sd_i2)

  lstm_a_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences = True))(lstm_a_1)



  max_pool_1 = tf.keras.layers.GlobalMaxPooling1D()(lstm_q_2)

  avg_pool_1 = tf.keras.layers.GlobalAveragePooling1D()(lstm_q_2)

  max_pool_2 = tf.keras.layers.GlobalMaxPooling1D()(lstm_a_2)

  avg_pool_2 = tf.keras.layers.GlobalAveragePooling1D()(lstm_a_2)



  hidden = tf.keras.layers.Concatenate()([max_pool_1, max_pool_2, avg_pool_1, avg_pool_2, i3, i4])

  dense = tf.keras.layers.Dense(256, activation = 'relu')(hidden)

  drop = tf.keras.layers.Dropout(rate = 0.15)(dense)

  

  out = tf.keras.layers.Dense(30, activation = 'sigmoid')(drop)



  model = tf.keras.Model(inputs = [i1, i2, i3, i4], outputs = [out])



  return model
myfold = np.random.randint(0,5, size = 1000)

myfold_counter = Counter(myfold)

print(myfold_counter)

most_common = myfold_counter.most_common(1)[0][0]

print(most_common)
tf.keras.backend.clear_session()

lr_sched = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3* (0.83 ** ((epoch+1) / 2)))



gkf = GroupKFold(n_splits = 5).split(X=train_df.url, groups = train_df.url)



for fold, (train_idx, valid_idx) in enumerate(gkf):

  prev_weights = None

  if fold != most_common:

    continue

  train_inputs = ((data['train_question_title'][train_idx], data['train_answer'][train_idx], data['train_question_title_use'][train_idx], 

                   data['train_answer_use'][train_idx]), (final_outputs[train_idx]))

  valid_inputs = (

                  (

                      ([data['train_question_title'][valid_idx]], [data['train_answer'][valid_idx]], [data['train_question_title_use'][valid_idx]], [data['train_answer_use'][valid_idx]]), 

                      (None)

                  )

               )

  BATCH_SIZE = 32

  train_dataset = get_generator(train_inputs)

  valid_dataset = tf.data.Dataset.from_tensor_slices(valid_inputs)



  model = create_model()

  optimizer = tf.keras.optimizers.Adam()

  model.compile(loss='binary_crossentropy', optimizer=optimizer)





  model.fit(train_dataset, epochs = 7, steps_per_epoch = train_idx.shape[0]//BATCH_SIZE,

            callbacks=[PredictCallback(valid_dataset, final_outputs[valid_idx]), lr_sched, EWA()])

  break

    
class Optimize:

    def __init__(self):

        self.clips = [[0, 1] for i in range(30)]

        self.ab_ = [(0, 0.15), (0.85, 1)]

        self.new_scores, self.scores = (None, None)

    def fit(self, labels, preds):

        self.scores = [SpearmanCorrCoeff(labels[:, i:i+1], preds[:, i:i+1]) for i in range(30)]

        for i in range(30):

            self.golden_section_search(labels[:, i:i+1], preds[:, i:i+1], i, 0)

            self.golden_section_search(labels[:, i:i+1], preds[:, i:i+1], i, 1)

        self.new_scores = [np.nan_to_num(SpearmanCorrCoeff(labels[:, i:i+1], np.clip(preds[:, i:i+1], self.clips[i][0], self.clips[i][1]))) for i in range(30)]

        for i in range(30):

            if self.scores[i] >= self.new_scores[i]:

                self.clips[i] = [0, 1]

    def golden_section_search(self, labels, preds, i, idx):

        (a, b) = self.ab_[idx]

        c = 0.618

        x1 = b - c*(b-a)

        x2 = (b-a)*c + a

        

        for epochs in range(10):

            self.clips[i][idx] = x1

            score_a = -self.score(labels, preds, i)

            self.clips[i][idx] = x2

            score_b = -self.score(labels, preds, i)

            if np.isnan(score_a):

                continue

            elif np.isnan(score_b):

                continue

            elif score_a <= score_b:

                b = x2

                x2 = x1

                x1 = b - c*(b-a)

            else:

                a = x1

                x1 = x2

                x2 = (b-a)*c + a

        

        self.clips[i][idx] = x1

        score_x1 = self.score(labels, preds, i)

        self.clips[i][idx] = x2

        score_x2 = self.score(labels, preds, i)

        if score_x1 > score_x2:

            self.clips[i][idx] = x1

        else:

            self.clips[i][idx] = x2

                    

            

    def score(self, labels, preds, i):

        return SpearmanCorrCoeff(labels, np.clip(preds, self.clips[i][0], self.clips[i][1]))

    def transform(self, preds):

        temp = preds.copy()

        for i in range(30):

            clipped = np.clip(preds[:, i], self.clips[i][0], self.clips[i][1])

            if np.unique(clipped).shape[0] > 1:

                temp[:, i][:] = clipped

        return temp
del train_inputs, valid_inputs, valid_dataset, train_dataset

gc.collect()
inputs = (

              (

                  ([data['train_question_title']], [data['train_answer']], [data['train_question_title_use']], [data['train_answer_use']]), 

                  (None)

              )

           )

dataset = tf.data.Dataset.from_tensor_slices(inputs)

predictions = model.predict(dataset)
train_y = final_outputs[train_idx]

valid_y = final_outputs[valid_idx]

train_preds = predictions[train_idx]

valid_preds = predictions[valid_idx]





opt = Optimize()

opt.fit(train_y, train_preds)

post_valid_preds = opt.transform(valid_preds)

print(f"Validation Score (Before) {SpearmanCorrCoeff(valid_y, valid_preds)}")

print(f"Validation Score (After) {SpearmanCorrCoeff(valid_y, post_valid_preds)}")
test_inputs = (

              (

                  ([data['test_question_title']], [data['test_answer']], [data['test_question_title_use']], [data['test_answer_use']]), 

                  (None)

              )

           )

test_dataset = tf.data.Dataset.from_tensor_slices(test_inputs)
test_preds = model.predict(test_dataset)

post_test_preds = opt.transform(test_preds)

submission = pd.read_csv('/kaggle/input/google-quest-challenge/sample_submission.csv')

submission.iloc[:,1:] = post_test_preds

submission.to_csv("submission.csv", index = False)

submission.head()