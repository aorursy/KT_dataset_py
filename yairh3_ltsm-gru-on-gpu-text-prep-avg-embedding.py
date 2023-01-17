import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

import gensim as gn

from tqdm import tqdm_notebook



from keras.layers import LSTM,Bidirectional,TimeDistributed, Embedding,Dense,Input,GlobalMaxPool1D,Flatten,Dropout

from keras.layers import CuDNNLSTM,CuDNNGRU,GlobalAveragePooling1D,GlobalMaxPooling1D,concatenate

from keras.models import Sequential,Model

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.callbacks import ModelCheckpoint

from keras import backend as K

from keras.optimizers import Adam

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints

import gc

# build attention layer 



class Attention(Layer):

    def __init__(self, step_dim,

                 W_regularizer=None, b_regularizer=None,

                 W_constraint=None, b_constraint=None,

                 bias=True, **kwargs):

        self.supports_masking = True

        self.init = initializers.get('glorot_uniform')



        self.W_regularizer = regularizers.get(W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)



        self.W_constraint = constraints.get(W_constraint)

        self.b_constraint = constraints.get(b_constraint)



        self.bias = bias

        self.step_dim = step_dim

        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

        

    def build(self, input_shape):

        assert len(input_shape) == 3



        self.W = self.add_weight((input_shape[-1],),

                                 initializer=self.init,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.W_regularizer,

                                 constraint=self.W_constraint)

        self.features_dim = input_shape[-1]



        if self.bias:

            self.b = self.add_weight((input_shape[1],),

                                     initializer='zero',

                                     name='{}_b'.format(self.name),

                                     regularizer=self.b_regularizer,

                                     constraint=self.b_constraint)

        else:

            self.b = None



        self.built = True

        

    def compute_mask(self, input, input_mask=None):

        return None



    def call(self, x, mask=None):

        

        features_dim = self.features_dim

        step_dim = self.step_dim



        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),

                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))



        if self.bias:

            eij += self.b



        eij = K.tanh(eij)



        a = K.exp(eij)



        if mask is not None:

            a *= K.cast(mask, K.floatx())



        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())



        a = K.expand_dims(a)

        weighted_input = x * a

        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):

        return input_shape[0], self.features_dim
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

df = pd.concat([df_train ,df_test])



print("Unsincere avg: ", df_train.target.mean())

print("total: ", df_train.target.count())

print('Test samples: ', df_test.qid.count())

df_train.head()
def add_lower(embedding, vocab):

    count = 0

    for word in vocab:

        if word in embedding and word.lower() not in embedding:  

            embedding[word.lower()] = embedding[word]

            count += 1

    print(f"Added {count} words to embedding")

    

def build_vocab(texts):

    sentences = texts.apply(lambda x: x.split()).values

    vocab = {}

    for sentence in sentences:

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab

vocab = build_vocab(df['question_text'])
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have",

                       "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", 

                       "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did",

                       "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 

                       "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 

                       "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 

                       "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", 

                       "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not",

                       "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", 

                       "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",

                       "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have",

                       "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would",

                       "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is",

                       "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",

                       "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",

                       "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", 

                       "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is",

                       "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will",

                       "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have",

                       "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", 

                       "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                       "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are",

                       "you've": "you have" }
def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-",

                 "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity',

                 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3',

                 'π': 'pi', }
def clean_special_chars(text, punct, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])

    

    for p in punct:

        text = text.replace(p, f' {p} ')

    

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last

    for s in specials:

        text = text.replace(s, specials[s])

    

    return text
mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',

                'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',

                'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 

                'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do',

                'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do',

                'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate',

                "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist',

                'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 

                'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 

                'demonitization': 'demonetization', 'demonetisation': 'demonetization', 'pokémon': 'pokemon'}
def correct_spelling(x, dic):

    for word in dic.keys():

        x = x.replace(word, dic[word])

    return x
# Lowering

df_train['treated_question'] = df_train['question_text'].apply(lambda x: x.lower())

df_test['treated_question'] = df_test['question_text'].apply(lambda x: x.lower())

# Contractions

df_train['treated_question'] = df_train['treated_question'].apply(lambda x: clean_contractions(x, contraction_mapping))

df_test['treated_question'] = df_test['treated_question'].apply(lambda x: clean_contractions(x, contraction_mapping))



# Special characters

df_train['treated_question'] = df_train['treated_question'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

df_test['treated_question'] = df_test['treated_question'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))



# Spelling mistakes

df_train['treated_question'] = df_train['treated_question'].apply(lambda x: correct_spelling(x, mispell_dict))

df_test['treated_question'] = df_test['treated_question'].apply(lambda x: correct_spelling(x, mispell_dict))
train, val = train_test_split(df_train, test_size=0.1, random_state=42)



n_words = 50000

tokenizer = Tokenizer(num_words=n_words)

tokenizer.fit_on_texts(list(train.treated_question))



q_train = tokenizer.texts_to_sequences(train.treated_question)

q_val = tokenizer.texts_to_sequences(val.treated_question)

q_test = tokenizer.texts_to_sequences(df_test.treated_question)



max_len = 100

q_train = pad_sequences(q_train,maxlen=max_len)

q_val = pad_sequences(q_val,maxlen=max_len)

q_test = pad_sequences(q_test,maxlen=max_len)



y_train = train.target

y_val = val.target



del train,val,df_train,df

gc.collect()
def f1(y_true, y_pred):

    def recall(y_true, y_pred):

        """Recall metric.



        Only computes a batch-wise average of recall.



        Computes the recall, a metric for multi-label classification of

        how many relevant items are selected.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        """Precision metric.



        Only computes a batch-wise average of precision.



        Computes the precision, a metric for multi-label classification of

        how many selected items are relevant.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))



#find the best threshold

def optim_thres(y_val,y_pred):

    score = 0

    thresholds = np.arange(0.1,0.501,0.01)

    for thres in thresholds:

        thres = np.round(thres,2)

        temp_pred = (y_pred > thres).astype(int)

        temp_score = f1_score(y_val,temp_pred)

        print("Thres: {} --------- F1: {}".format(thres,temp_score))

        if temp_score > score:

            score = temp_score

            final_thres = thres

    return final_thres
# try to mix all embeddings 

# first read & fit a model with all of them 



emb_file = "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"

glove_dic = {}

for line in tqdm_notebook(open(emb_file)):

    temp = line.split(" ")

    glove_dic[temp[0]] = np.asarray(temp[1:],dtype='float32')

add_lower(glove_dic, vocab)

word_index = tokenizer.word_index

emb_size = glove_dic['.'].shape[0]

emb_matrix = np.zeros((n_words,emb_size))

for w,index in word_index.items():

    if index >= n_words:

        continue

    vec = glove_dic.get(w)

    if vec is not None:

        emb_matrix[index,:] = vec

inp = Input(shape=(max_len,))

x = Embedding(input_dim=n_words,output_dim=emb_size, weights=[emb_matrix],trainable=False)(inp)

x1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)

x2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)

x1 = GlobalAveragePooling1D()(x1)

x2 = GlobalAveragePooling1D()(x2)

concat = concatenate([x1, x2])

x = Dense(64, activation="relu")(concat)

drop = Dropout(0.1)(x)

x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inp,output=x)

model.summary()

model_name = 'lstm_glove_emb'

checkpoint = ModelCheckpoint(filepath='./{}.hdf5'.format(model_name),

                             monitor='val_loss',mode='min',verbose=1,

                            save_best_only=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc',f1])

history  = model.fit(q_train,y_train,batch_size=1500,epochs=5,

                     validation_data=(q_val,y_val),verbose=1,callbacks=[checkpoint])
del glove_dic

gc.collect()

model.load_weights('./{}.hdf5'.format(model_name))

y_pred_glove = model.predict(q_val,batch_size=1064,verbose=1)

pred_glove = model.predict(q_test,batch_size=1064,verbose=1)

emb_file = "../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt"

emb_dic = {}

for line in tqdm_notebook(open(emb_file,encoding='utf-8',errors='ignore')):

    temp = line.split(" ")

    emb_dic[temp[0]] = np.asarray(temp[1:],dtype='float32')

    

emb_matrix = np.zeros((n_words,emb_size))

for w,index in word_index.items():

    if index >= n_words:

        continue

    vec = emb_dic.get(w)

    if vec is not None:

        emb_matrix[index,:] = vec
inp = Input(shape=(max_len,))

x = Embedding(input_dim=n_words,output_dim=emb_size, weights=[emb_matrix],trainable=False)(inp)

x1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)

x2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)

x1 = GlobalAveragePooling1D()(x1)

x2 = GlobalAveragePooling1D()(x2)

concat = concatenate([x1, x2])

x = Dense(64, activation="relu")(concat)

drop = Dropout(0.1)(x)

x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inp,output=x)

model.summary()

model_name = 'lstm_paragram_emb'

checkpoint = ModelCheckpoint(filepath='./{}.hdf5'.format(model_name),

                             monitor='val_loss',mode='min',verbose=1,

                            save_best_only=True)


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc',f1])

history  = model.fit(q_train,y_train,batch_size=1500,epochs=5,

                     validation_data=(q_val,y_val),verbose=1,callbacks=[checkpoint])
del emb_dic

gc.collect()

model.load_weights('./{}.hdf5'.format(model_name))

y_pred_paragram = model.predict(q_val,batch_size=1064,verbose=1)

pred_paragram = model.predict(q_test,batch_size=1064,verbose=1)

emb_file = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

emb_dic = {}

for line in tqdm_notebook(open(emb_file,encoding='utf-8',errors='ignore')):

    temp = line.split(" ")

    emb_dic[temp[0]] = np.asarray(temp[1:],dtype='float32')

    

emb_matrix = np.zeros((n_words,emb_size))

for w,index in word_index.items():

    if index >= n_words:

        continue

    vec = emb_dic.get(w)

    if vec is not None:

        emb_matrix[index,:] = vec
inp = Input(shape=(max_len,))

x = Embedding(input_dim=n_words,output_dim=emb_size, weights=[emb_matrix])(inp)

x1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)

x2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)

x1 = GlobalAveragePooling1D()(x1)

x2 = GlobalAveragePooling1D()(x2)

concat = concatenate([x1, x2])

x = Dense(64, activation="relu")(concat)

drop = Dropout(0.1)(x)

x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inp,output=x)

model.summary()

model_name = 'lstm_wiki_emb'

checkpoint = ModelCheckpoint(filepath='./{}.hdf5'.format(model_name),

                             monitor='val_loss',mode='min',verbose=1,

                            save_best_only=True)


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc',f1])

history  = model.fit(q_train,y_train,batch_size=1500,epochs=5,

                     validation_data=(q_val,y_val),verbose=1,callbacks=[checkpoint])
del emb_dic

gc.collect()

model.load_weights('./{}.hdf5'.format(model_name))

y_pred_wiki = model.predict(q_val,batch_size=1064,verbose=1)

pred_wiki = model.predict(q_test,batch_size=1064,verbose=1)
y_pred_all = np.mean(np.array([y_pred_glove,y_pred_paragram,y_pred_wiki]),axis=0)

print(y_pred_all.shape)

final_thresh = optim_thres(y_val,y_pred_all)
final_pred = np.mean(np.array([pred_glove,pred_paragram,pred_wiki]),axis=0)

sub_pred = (final_pred > final_thresh).astype(int)

sub = pd.DataFrame({"qid":df_test["qid"].values})

sub['prediction'] = sub_pred

sub.to_csv("submission.csv", index=False)