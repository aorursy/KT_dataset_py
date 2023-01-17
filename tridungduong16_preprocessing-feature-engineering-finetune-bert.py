!pip install ../input/sacremoses/sacremoses-master/ > /dev/null



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import tensorflow_hub as hub

import tensorflow as tf

import tensorflow.keras.backend as K

import gc

import sys

import os

import sys

import glob

import torch

import re 

import math

import pickle

import datetime

import string 

import nltk 

import spacy

import tensorflow.keras.backend as K



from scipy.stats import spearmanr

from math import floor, ceil

from tqdm.notebook import tqdm

from sklearn.model_selection import GroupKFold

from scipy import spatial

from nltk.tokenize import sent_tokenize

from nltk import wordpunct_tokenize

from sklearn.linear_model import MultiTaskElasticNet

from scipy.stats import spearmanr, rankdata



sys.path.insert(0, "../input/transformers/transformers-master/")



import transformers as ppb





DEVICE = torch.device("cuda")
root_path = '../input/google-quest-challenge/'

ss = pd.read_csv(root_path + '/sample_submission.csv')

train = pd.read_csv(root_path + '/train.csv')

test = pd.read_csv(root_path + '/test.csv')
# train.columns


# train['full_text']
# technology=train[train.category == "TECHNOLOGY"]
train[['question_title', 'question_body', 'answer']]
train['question_title'] = train['question_title'] + '?'

train['question_body'] = train['question_body'] + '?'

train['answer'] = train['answer'] + '.'

train['full_question'] = train['question_title'] + " [SEP] " + train['question_body']

test['full_question'] = test['question_title'] + " [SEP] " + test['question_body']
# count = 0

# for i in train.answer:

#   print(count)

#   print(i)

#   print("-"*100)

#   count += 1

#   if count == 10:

#     break
DEVICE = torch.device("cuda")
bert_model_config = '../input/pretrained-bert-models-for-pytorch/bert-base-uncased/bert_config.json'

bert_config = ppb.BertConfig.from_json_file(bert_model_config)

tokenizer = ppb.BertTokenizer.from_pretrained('../input/pretrained-bert-models-for-pytorch/bert-base-uncased-vocab.txt')

bert_model = ppb.BertModel.from_pretrained('../input/pretrained-bert-models-for-pytorch/bert-base-uncased/', config=bert_config)

bert_model.to(DEVICE)
#text = 'i love you embedding'

#print(tokenizer.tokenize(text))

#print(tokenizer.vocab)
# -*- coding: utf-8 -*-

import re

alphabets= "([A-Za-z])"

prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"

suffixes = "(Inc|Ltd|Jr|Sr|Co)"

starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"

acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"

websites = "[.](com|net|org|io|gov)"



def split_into_sentences(text):

    text = " " + text + "  "

    text = text.replace("\n"," ")

    text = re.sub(prefixes,"\\1<prd>",text)

    text = re.sub(websites,"<prd>\\1",text)

    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")

    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)

    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)

    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)

    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)

    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)

    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)

    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)

    if "”" in text: text = text.replace(".”","”.")

    if "\"" in text: text = text.replace(".\"","\".")

    if "!" in text: text = text.replace("!\"","\"!")

    if "?" in text: text = text.replace("?\"","\"?")

    text = text.replace(".",".<stop>")

    text = text.replace("?","?<stop>")

    text = text.replace("!","!<stop>")

    text = text.replace("<prd>",".")

    sentences = text.split("<stop>")

    sentences = sentences[:-1]

    sentences = [s.strip() for s in sentences]

    return sentences

import itertools

words = set(nltk.corpus.words.words())



def remove_non_english(text):

    return " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in words or not w.isalpha())



#     doc = spacy_nlp(x) 

#     tokens = [token.text for token in doc]

#     preprocessed_doc = " ".join(w for w in tokens if w.lower() in words)

#     return preprocessed_doc





def add_token_url(text):

    URL_REGEX = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""

    urls = re.findall(URL_REGEX,text)

    count = 0

    for url in urls:

        text = text.replace(url, '<URL>')

    text = sent_tokenize(text)

    text = [x for x in text if x not in string.punctuation]

    result = []

    text = [x.splitlines() for x in text]

    text = list(itertools.chain.from_iterable(text))

    text = list(filter(None, text))



    text = [remove_non_english(x) for x in text]

    text = [x for x in text if x not in string.punctuation]

    text = [re.sub(r'[^\w\s]','',x) for x in text]

    text = [re.sub(' +', ' ', x) for x in text]

    text = [x.strip() for x in text]

    text = list(filter(None, text))



    return ' [SEP] '.join(text)



train['preprocessed_full_question'] = [add_token_url(x) for x in train['full_question']]

train['preprocessed_answer'] = [add_token_url(x) for x in train['answer']]



test['preprocessed_full_question'] = [add_token_url(x) for x in test['full_question']]

test['preprocessed_answer'] = [add_token_url(x) for x in test['answer']]

print(len(tokenizer))  # 28997

tokenizer.add_tokens(["<URL>"])

print(len(tokenizer))  # 28997



bert_model.resize_token_embeddings(len(tokenizer)) 

def convert_text_to_vector(df, col, tokenizer, model):

    df[col] = [x[:512] for x in df[col]]

    tokenized = df[col].apply(lambda x: tokenizer.encode(x, add_special_tokens = True))

    max_len= 512 

    padded = [i + [0]*(max_len - len(i)) for i in tokenized]



    for i in tqdm(range(len(tokenized))):

        tokenized[i].extend([0]*(max_len - len(tokenized[i])))

    tokenized = [np.array(x) for x in tokenized]

    tokenized = np.array(tokenized)

    attention_mask = np.where(tokenized != 0,1,0)

    input_ids = torch.tensor(tokenized).to(DEVICE)

    attention_mask = torch.tensor(attention_mask).to(DEVICE)

    

    segments = []

    for tokens in tqdm(tokenized):

      segment = []

      current_segment_id = 0

      for token in tokens:

          segment.append(current_segment_id)

          if token == 102:

            current_segment_id += 1

      segment = segment + [current_segment_id+1] * (512 - len(tokens))

      segments.append(segment)

    segments = torch.tensor(segments).to(DEVICE)

    return input_ids, attention_mask, segments

    



batch_size = 64



targets = [

        'question_asker_intent_understanding',

        'question_body_critical',

        'question_conversational',

        'question_expect_short_answer',

        'question_fact_seeking',

        'question_has_commonly_accepted_answer',

        'question_interestingness_others',

        'question_interestingness_self',

        'question_multi_intent',

        'question_not_really_a_question',

        'question_opinion_seeking',

        'question_type_choice',

        'question_type_compare',

        'question_type_consequence',

        'question_type_definition',

        'question_type_entity',

        'question_type_instructions',

        'question_type_procedure',

        'question_type_reason_explanation',

        'question_type_spelling',

        'question_well_written',

        'answer_helpful',

        'answer_level_of_information',

        'answer_plausible',

        'answer_relevance',

        'answer_satisfaction',

        'answer_type_instructions',

        'answer_type_procedure',

        'answer_type_reason_explanation',

        'answer_well_written'    

    ]



y = train[targets].values

# ----
def chunks(l, n):

    """Yield successive n-sized chunks from l."""

    for i in range(0, len(l), n):

        yield l[i:i + n]



def splitDataFrameIntoSmaller(df, chunkSize = 10000): 

    listOfDf = list()

    numberChunks = len(df) // chunkSize + 1

    for i in range(numberChunks):

        listOfDf.append(df[i*chunkSize:(i+1)*chunkSize])

    return listOfDf



import sklearn

le = sklearn.preprocessing.LabelEncoder()

le.fit(train['category'])

category_features_train = le.fit_transform(train['category'])

category_features_test = le.fit_transform(test['category'])

module_url = "../input/universalsentenceencoderlarge4/"

embed = hub.load(module_url)



def encoding_sentence(df, col, batch_size, model):   

    all_features = []

    for tokenized_batch in tqdm(splitDataFrameIntoSmaller(df[col].values, chunkSize = batch_size)):

        all_features.append(model(tokenized_batch)["outputs"].numpy())

    all_features = np.vstack(all_features)

    return all_features



def calculate_text_distance(question_title_features,question_body_features,answer_features):



    dist1 = list(map(lambda x, y: np.linalg.norm(x-y), question_title_features, question_body_features))

    dist2 = list(map(lambda x, y: np.linalg.norm(x-y), question_body_features,answer_features))

    dist3 =list(map(lambda x, y: np.linalg.norm(x-y), answer_features,question_title_features))

    cosdist = np.array([dist1, dist2, dist3])

    cosdist = cosdist.T



    dist1 = list(map(lambda x, y: spatial.distance.cosine(x,y), question_title_features, question_body_features))

    dist2 = list(map(lambda x, y: spatial.distance.cosine(x,y), question_body_features,answer_features))

    dist3 = list(map(lambda x, y: spatial.distance.cosine(x,y), answer_features,question_title_features))

    l2dist = np.array([dist1, dist2, dist3])

    l2dist = l2dist.T



    distance = np.hstack([cosdist,l2dist])

    return distance









question_title_encoding = encoding_sentence(train, 'question_title', 32, embed)

question_body_encoding = encoding_sentence(train, 'question_body', 32, embed)

answer_encoding  = encoding_sentence(train, 'answer', 32, embed)



question_title_encoding_test = encoding_sentence(test, 'question_title', 32, embed)

question_body_encoding_test = encoding_sentence(test, 'question_body', 32, embed)

answer_encoding_test  = encoding_sentence(test, 'answer', 32, embed)



train_distance = calculate_text_distance(question_title_encoding,question_body_encoding,answer_encoding)

test_distance = calculate_text_distance(question_title_encoding_test,question_body_encoding_test,answer_encoding_test)









def compute_spearmanr(trues, preds):    

    rhos = []

    for col_trues, col_pred in zip(trues.T, preds.T):

        rhos.append(

            spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)

    return np.mean(rhos)





class CustomCallback(tf.keras.callbacks.Callback):

    

    def __init__(self, valid_data, test_data, batch_size=16, fold=None):



        self.valid_inputs = valid_data[0]

        self.valid_outputs = valid_data[1]

        self.test_inputs = test_data

        

        self.batch_size = batch_size

        self.fold = fold

        

    def on_train_begin(self, logs={}):

        self.valid_predictions = []

        self.test_predictions = []

        

    def on_epoch_end(self, epoch, logs={}):

        self.valid_predictions = self.model.predict(self.valid_inputs)

        

        rho_val = compute_spearmanr(self.valid_outputs, self.valid_predictions)

        print('\n Epoch {}, Validation score {}'.format(epoch,rho_val))



        

        if self.fold is not None:

            self.model.save_weights(f'bert-base-{fold}-{epoch}.h5py')

        

        #self.test_predictions = self.model.predict(self.test_inputs)

# BERT_PATH = '../input/bert-base-uncased-huggingface-transformer/'

# from transformers import *



class BertClassification(tf.keras.Model):

    def __init__(self,flag_distance = False, flag_cat = False,flag_lstm = False, trainable = True):

        super().__init__(name='BertClassification')

        self.bert_layer = hub.KerasLayer('../input/bert-base-from-tfhub/bert_en_uncased_L-12_H-768_A-12', trainable=trainable)

#         config = BertConfig() # print(config) to see settings

#         config.output_hidden_states = False # Set to True to obtain hidden states

#         config.trainable = True

#         self.bert_layer = TFBertModel.from_pretrained(BERT_PATH+'bert-base-uncased-tf_model.h5', config=config)



        self.global_avarage = tf.keras.layers.GlobalAveragePooling1D()

        self.dense_out = tf.keras.layers.Dense(30, activation="sigmoid", name="dense_output")

        self.embed =  tf.keras.layers.Embedding(500, 64, input_length=1)

        self.dropout = tf.keras.layers.Dropout(0.25)

        self.flag_distance = flag_distance

        self.flag_cat = flag_cat

        self.flag_lstm = flag_lstm





    def call(self, inputs):

        max_len = 512

        inputs = [tf.cast(x, tf.int32) for x in inputs]



        input_word_ids_title, input_masks_title, input_segments_title = inputs[0],inputs[1],inputs[2]

        input_word_ids_answer, input_masks_answer, input_segments_answer =  inputs[3],inputs[4],inputs[5]     



        features_cat = inputs[6]

        distance_features = tf.cast(inputs[7], tf.float32)  



        _, sequence_output_title = self.bert_layer([input_word_ids_title, input_masks_title, input_segments_title])

        global_title = self.global_avarage(sequence_output_title)





        _, sequence_output_answer = self.bert_layer([input_word_ids_answer, input_masks_answer, input_segments_answer])

        global_answer = self.global_avarage(sequence_output_answer)





        embedding_cat = self.embed(features_cat)

        embedding_cat = self.global_avarage(embedding_cat)

        embedding_cat = self.dropout(embedding_cat)

        distance_features = self.dropout(distance_features)



        concat = tf.keras.layers.concatenate([global_title,

                                              global_answer,

                                              embedding_cat,

                                              distance_features])



        concat = self.dropout(concat)

        out = self.dense_out(concat)

        return out



# model = BertClassification()
# def training(X_train,y_train,X_val,y_val,X_test):  

#   batch_size  =  2

#   custom_callback = CustomCallback(valid_data=(X_val,y_val),test_data=X_test, batch_size=batch_size)

#   learning_rate = 3e-5

#   epochs = 20

#   loss_function = 'binary_crossentropy'

#   optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

#   model = BertClassification()

#   model.compile(loss=loss_function, optimizer=optimizer)

#   model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,callbacks=[custom_callback])

#   return model





# def training_2(X_train,y_train,X_val,y_val,X_test):  

#   batch_size  =  2

#   custom_callback = CustomCallback(valid_data=(X_val,y_val),test_data=X_test, batch_size=batch_size)

#   learning_rate = 3e-5

#   epochs = 3

#   loss_function = 'binary_crossentropy'

#   optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

#   model = question_answer_model()

#   model.compile(loss=loss_function, optimizer=optimizer)

#   model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,callbacks=[custom_callback])

#   return model  
#inputs_ids, attention_masks, segments = convert_text_to_vector(train, 'full_text', tokenizer, bert_model)

#inputs_ids_test, attention_masks_test, segments_test = convert_text_to_vector(test, 'full_text', tokenizer, bert_model)





inputs_ids_title, attention_masks_title, segments_title = convert_text_to_vector(train, 'preprocessed_full_question', tokenizer, bert_model)

inputs_ids_test_title, attention_masks_test_title, segments_test_title = convert_text_to_vector(test, 'preprocessed_full_question', tokenizer, bert_model)



inputs_ids_answer, attention_masks_answer, segments_answer = convert_text_to_vector(train, 'preprocessed_answer', tokenizer, bert_model)

inputs_ids_test_answer, attention_masks_test_answer, segments_test_answer = convert_text_to_vector(test, 'preprocessed_answer', tokenizer, bert_model)







X = [inputs_ids_title.cpu().data.numpy(), 

     attention_masks_title.cpu().data.numpy(), 

     segments_title.cpu().data.numpy(),

     inputs_ids_answer.cpu().data.numpy(),

     attention_masks_answer.cpu().data.numpy(),

     segments_answer.cpu().data.numpy(),

     category_features_train,

     train_distance

     ]



X_test = [inputs_ids_test_title.cpu().data.numpy(), 

     attention_masks_test_title.cpu().data.numpy(), 

     segments_test_title.cpu().data.numpy(),

     inputs_ids_test_answer.cpu().data.numpy(),

     attention_masks_test_answer.cpu().data.numpy(),

     segments_test_answer.cpu().data.numpy(),

     category_features_test,

     test_distance

     ]





import timeit



batch_size  = 4

learning_rate = 3e-5

epochs = 3

loss_function = 'binary_crossentropy'



gkf = GroupKFold(n_splits=5).split(X=train.category, groups=train.category)



valid_preds = []

test_preds = []

validation_score = []





for fold, (train_idx, valid_idx) in tqdm(enumerate(gkf)):

    if fold in [1, 2]:

        print("Fold {}".format(fold))



        start = timeit.default_timer()



        X_train = [X[i][train_idx] for i in range(len(X))]

        y_train = y[train_idx]

        X_val = [X[i][valid_idx] for i in range(len(X))]

        y_val = y[valid_idx]   

        K.clear_session()

        

        custom_callback = CustomCallback(valid_data=(X_val,y_val),test_data=X_test, batch_size=batch_size)

        

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        

        model = BertClassification()

        model.compile(loss=loss_function, optimizer=optimizer)



        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        

        test_prediction = model.predict(X_test)

        valid_preds.append(model.predict(X_val))

        test_preds.append(test_prediction)       

        rho_val = compute_spearmanr(y_val, valid_preds[-1])

        validation_score.append(rho_val)

        

        #print("Spearman score {}".format(rho_val))

        

        stop = timeit.default_timer()

        training_time = stop  -  start  

        

        print("Training time {}".format(training_time))

        

        

        del model

        del X_train

        del y_train

        del X_val

        del y_val 



print("Validation score {}".format(np.mean(validation_score)))



test_preds[0]
test_preds[1].shape
np.mean(test_preds, axis=0)
"""





from sklearn.model_selection import train_test_split

X_train,y_train, X_val, y_val = train_test_split(X,y,random_state = 1,test_size = 0.25)

print("Validation score {}".format(compute_spearmanr(y_val, model.predict(X_val)))

"""

# test_preds[0]
# batch_size  = 2

# learning_rate = 3e-5

# epochs = 2

# loss_function = 'binary_crossentropy'



# X_val = [X[i][:500] for i in range(len(X))]

# y_val = y[:500]   

        



# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# model = BertClassification()

# model.compile(loss=loss_function, optimizer=optimizer)



# # custom_callback = CustomCallback(valid_data=(X_val,y_val),test_data=X_test, batch_size=batch_size)

# model.fit(X, y, epochs=epochs, batch_size=batch_size)

    

train['category'].values
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder()

onehotencoder.fit(category_features_train.reshape(-1,1))

    

category_onehot_train = onehotencoder.transform(category_features_train.reshape(-1,1)).toarray()

category_onehot_test = onehotencoder.transform(category_features_test.reshape(-1,1)).toarray()
category_onehot_train.shape
train_distance.shape
def sigmoid(X):

    return 1/(1+np.exp(-X))
X = np.hstack([

     category_onehot_train,

     train_distance

     ])



X_test = np.hstack([

     category_onehot_test,

     test_distance

     ])



elastic_model = MultiTaskElasticNet(alpha=0.001, random_state=42, l1_ratio=0.5)

elastic_model.fit(X, y)

elastic_prediction = sigmoid(elastic_model.predict(X_test))
elastic_prediction
test_preds.append(elastic_prediction)
test_preds = np.mean(test_preds, axis=0)
test_preds
submission = pd.DataFrame(columns = list(ss.columns))

submission['qa_id'] = test['qa_id']

submission[targets] = test_preds

submission.to_csv("submission.csv", index = False)
submission
# https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/