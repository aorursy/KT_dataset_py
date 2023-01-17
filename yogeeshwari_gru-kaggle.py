!pip install py7zr
%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

import scipy

import sqlite3

import pandas as pd

import numpy as np

import nltk

import string

import datetime

import time

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb

import gc

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

import shutil

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from sklearn.linear_model import SGDRegressor

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_squared_error

from py7zr import unpack_7zarchive

import math

import re

# Tutorial about Python regular expressions: https://pymotw.com/2/re/

import string

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from scipy.sparse import csr_matrix

from scipy.sparse import hstack

from sklearn.preprocessing import StandardScaler

from scipy.sparse import coo_matrix, hstack

from gensim.models import Word2Vec

from gensim.models import KeyedVectors

from prettytable import PrettyTable

from sklearn.linear_model import RidgeCV

import pickle

import zipfile

from tqdm import tqdm

import os

from sklearn.linear_model import Ridge

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from stop_words import get_stop_words

from collections import Counter

import tensorflow as tf

from tensorflow.keras.callbacks import TensorBoard,EarlyStopping,ModelCheckpoint

from tensorflow.keras.layers import Input, Embedding, GRU, Dense,Flatten

from tensorflow.keras.models import Model,load_model

from numpy import zeros

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import plot_model

from contextlib import contextmanager

tf.keras.backend.clear_session()
def extract_from_archive():

    #https://stackoverflow.com/questions/50745486/how-to-use-pyunpack-to-unpack-7z-file

    if not os.path.exists('/kaggle/working/train/'):

        os.makedirs('/kaggle/working/train/')

    shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)

    shutil.unpack_archive('/kaggle/input/mercari-price-suggestion-challenge/train.tsv.7z', '/kaggle/working/train/')

    shutil.unpack_archive('/kaggle/input/mercari-price-suggestion-challenge/test_stg2.tsv.zip', '/kaggle/working/test/')



extract_from_archive()
def split_categories(category):

    '''

    function that splits the category column in the dataset and creates 3 new columns:

    'main_category','sub_cat_1','sub_cat_2'

    '''

    try:

        sub_cat_1,sub_cat_2,sub_cat_3 = category.split("/")

        return sub_cat_1,sub_cat_2,sub_cat_3

    except:

        return ("No label","No label","No label")



def create_split_categories(data):

    '''

    function that creates 3 new columns using split_categories function

    : 'main_category','sub_cat_1','sub_cat_2'

    '''

    data['main_category'],data['sub_cat_1'],data['sub_cat_2']=zip(*data['category_name'].\

                                                                  apply(lambda x: split_categories(x)))



def log_price(price):

    return np.log1p(price)#changes



#https://www.kaggle.com/valkling/mercari-rnn-2ridge-models-with-notes-0-42755

def countwords(text):

    try:

        if text == 'No description yet':

            return 0

        else:

            text = text.lower()

            words = [w for w in text.split(" ")]

            return len(words)

    except: 

        return 0



def fill_nan(dataset):

    '''

    Function to fill the NaN values in various columns

    '''

    dataset["item_description"].fillna("No description yet",inplace=True)

    dataset["brand_name"].fillna("missing",inplace=True)

    dataset["category_name"].fillna("missing",inplace=True)



def get_dummies_item_id_shipping(df):

    df['item_condition_id'] = df["item_condition_id"].astype("category")

    df['shipping'] = df["shipping"].astype("category")

    item_id_shipping = csr_matrix(pd.get_dummies(df[['item_condition_id', 'shipping']],\

                                                      sparse=True).values)

    return item_id_shipping



def one_hot_encode(train,test):

    '''

    Function to one hot encode the categorical columns

    '''

    vectorizer = CountVectorizer(token_pattern='.+')

    

    vectorizer = vectorizer.fit(train['category_name'].values) # fit has to happen only on train data

    column_cat = vectorizer.transform(test['category_name'].values)



    #vectorizing the main_category column

    vectorizer = vectorizer.fit(train['main_category'].values) # fit has to happen only on train data

    column_mc = vectorizer.transform(test['main_category'].values)

    

    #vectorizing sub_cat_1 column

    vectorizer = vectorizer.fit(train['sub_cat_1'].values) # fit has to happen only on train data

    column_sb1 = vectorizer.transform(test['sub_cat_1'].values)

    

    #vectorizing sub_cat_2 column

    vectorizer = vectorizer.fit(train['sub_cat_2'].values) # fit has to happen only on train data

    column_sb2 = vectorizer.transform(test['sub_cat_2'].values)



    #vectorizing brand column

    vectorizer = vectorizer.fit(train['brand_name'].astype(str)) # fit has to happen only on train data

    brand_encodes = vectorizer.transform(test['brand_name'].astype(str))



    #https://github.com/numpy/numpy/issues/11666

   # vectorizer = CountVectorizer(vocabulary= list(train['shipping'].unique()),binary = True)

    #shipping

    #vectorizer = vectorizer.fit(train['shipping'].astype(str)) # fit has to happen only on train data

    #column_shipping = vectorizer.transform(test['shipping'].astype(str))

    

    #vectorizer = CountVectorizer(vocabulary= list(train['item_condition_id'].unique()),binary = True)

    #item_condition_id

    #vectorizer = vectorizer.fit(train['item_condition_id'].astype(str)) # fit has to happen only on train data

    #column_item_id = vectorizer.transform(test['item_condition_id'].astype(str))

    

    print("created OHE columns for main_category,sub_cat_1,sub_cat_2\n")

    print(column_cat.shape)

    print(column_mc.shape)

    print(column_sb1.shape)

    print(column_sb2.shape)

    print(brand_encodes.shape)

    #print(column_shipping.shape)

    #print(column_item_id.shape)

    print("="*100)

    return column_cat,column_mc,column_sb1,column_sb2,brand_encodes



def rank_category(dataset,column_name):

    '''This function takes a column name which is categorical and returns the categories with rank'''

    counter = dataset[column_name].value_counts().index.values

    total = list(dataset[column_name])

    ranked_cat = {}

    for i in range(1,len(counter)+1):

        ranked_cat.update({counter[i-1] : i})

    return ranked_cat,len(counter)



def encode_ranked_category(train,test,column):

    '''

    This function calls the rank_category function and returns the encoded category column    '''

    train[column] = train[column].astype('category')

    test[column] = test[column].astype('category')

    

    cat_list = list(train[column].unique())

    ranked_cat_tr,count = rank_category(train,column)



    encoded_col_tr = []

    encoded_col_te = []



    for category in train[column]:

        encoded_col_tr.append(ranked_cat_tr[category])



    for category in test[column]:

        if category in cat_list:

            encoded_col_te.append(ranked_cat_tr[category])

        else:

            encoded_col_te.append(0)

    

    encoded_col_tr = np.asarray(encoded_col_tr)

    encoded_col_te = np.asarray(encoded_col_te)

    return encoded_col_tr,encoded_col_te,count



def tokenize_text(train,test,column):

    global t

    t = Tokenizer()

    t.fit_on_texts(train[column].str.lower())

    vocab_size = len(t.word_index) + 1

    # integer encode the documents

    encoded_text_tr = t.texts_to_sequences(train[column].str.lower())

    encoded_text_te = t.texts_to_sequences(test[column].str.lower())

    return encoded_text_tr,encoded_text_te,vocab_size



def data_gru(train,test):

    

    global max_length,desc_size,name_size

    encoded_brand_tr,encoded_brand_te,brand_len = encode_ranked_category(train,test,'brand_name')

    #encoded_cat_tr,encoded_cat_te,cat_len = encode_ranked_category(train,test,'category_name')

    encoded_main_cat_tr,encoded_main_cat_te,main_cat_len = encode_ranked_category(train,test,'main_category')

    encoded_sub_cat_1_tr,encoded_sub_cat_1_te,sub_cat1_len = encode_ranked_category(train,test,'sub_cat_1')

    encoded_sub_cat_2_tr,encoded_sub_cat_2_te,sub_cat2_len = encode_ranked_category(train,test,'sub_cat_2')

    

    tokenized_desc_tr,tokenized_desc_te,desc_size = tokenize_text(train,test,'item_description')

     

    tokenized_name_tr,tokenized_name_te,name_size = tokenize_text(train,test,'name')

      

    max_length = 160

    desc_tr_padded = pad_sequences(tokenized_desc_tr, maxlen=max_length, padding='post')

    desc_te_padded = pad_sequences(tokenized_desc_te, maxlen=max_length, padding='post')

    del tokenized_desc_tr,tokenized_desc_te



    name_tr_padded = pad_sequences(tokenized_name_tr, maxlen=10, padding='post')

    name_te_padded = pad_sequences(tokenized_name_te, maxlen=10, padding='post')

    del tokenized_name_tr,tokenized_name_te



    gc.collect()



    train_inputs = [name_tr_padded,desc_tr_padded,encoded_brand_tr.reshape(-1,1),\

                    encoded_main_cat_tr.reshape(-1,1),encoded_sub_cat_1_tr.reshape(-1,1),\

                    encoded_sub_cat_2_tr.reshape(-1,1),train['shipping'],\

                    train['item_condition_id'],train['wc_desc'],\

                    train['wc_name']]

    test_inputs = [name_te_padded,desc_te_padded,encoded_brand_te.reshape(-1,1),\

                    encoded_main_cat_te.reshape(-1,1),encoded_sub_cat_1_te.reshape(-1,1),\

                    encoded_sub_cat_2_te.reshape(-1,1),test['shipping'],\

                    test['item_condition_id'],test['wc_desc'],\

                    test['wc_name']]

    

    item_condition_counter = train['item_condition_id'].value_counts().index.values



    list_var = [brand_len,main_cat_len,sub_cat1_len,sub_cat2_len,len(item_condition_counter)]

    

    return train_inputs,test_inputs,list_var



def construct_GRU(train,var_list,drop_out_list):

    #GRU input layer for name

    input_name =  tf.keras.layers.Input(shape=(10,), name='name')

    embedding_name = tf.keras.layers.Embedding(name_size, 20)(input_name)

    gru_name = tf.keras.layers.GRU(8)(embedding_name)

    #flatten1 = tf.keras.layers.Flatten()(lstm_out)



    #GRU input layer for description

    input_desc =  tf.keras.layers.Input(shape=(max_length,), name='desc')

    embedding_desc = tf.keras.layers.Embedding(desc_size, 60)(input_desc)

    gru_desc = tf.keras.layers.GRU(16)(embedding_desc)



    #input layer for brand_name

    input_brand =  tf.keras.layers.Input(shape=(1,), name='brand')

    embedding_brand = tf.keras.layers.Embedding(var_list[0] + 1, 10)(input_brand)

    flatten1 = tf.keras.layers.Flatten()(embedding_brand)



    #categorical input layer main_category

    input_cat = tf.keras.layers.Input(shape=(1,), name='main_cat')

    Embed_cat = tf.keras.layers.Embedding(var_list[1] + 1, \

                                          10,input_length=1)(input_cat)

    flatten2 = tf.keras.layers.Flatten()(Embed_cat)



    #categorical input layer sub_cat_1

    input_subcat1 = tf.keras.layers.Input(shape=(1,), name='subcat1')

    Embed_subcat1 = tf.keras.layers.Embedding(var_list[2] + 1, \

                                              10,input_length=1)(input_subcat1)

    flatten3 = tf.keras.layers.Flatten()(Embed_subcat1)



    #categorical input layer sub_cat_2

    input_subcat2 = tf.keras.layers.Input(shape=(1,), name='subcat2')

    Embed_subcat2 = tf.keras.layers.Embedding(var_list[3] + 1, \

                                              10,input_length=1)(input_subcat2)

    flatten4 = tf.keras.layers.Flatten()(Embed_subcat2)



    #categorical input layer shipping

    input_shipping = tf.keras.layers.Input(shape=(1,), name='shipping')

    # Embed_shipping = tf.keras.layers.Embedding(var_list[3] + 1, \

    #                                            7,input_length=1)(input_shipping)

    # flatten5 = tf.keras.layers.Flatten()(Embed_shipping)



    #categorical input layer item_condition_id

    input_item = tf.keras.layers.Input(shape=(1,), name='item_condition_id')

    Embed_item = tf.keras.layers.Embedding(var_list[4] + 1, \

                                           5,input_length=1)(input_item)

    flatten5 = tf.keras.layers.Flatten()(Embed_item)



    #numerical input layer

    desc_len_input = tf.keras.layers.Input(shape=(1,), name='description_length')

    desc_len_embd = tf.keras.layers.Embedding(DESC_LEN,5)(desc_len_input)

    flatten6 = tf.keras.layers.Flatten()(desc_len_embd)



    #name_len input layer

    name_len_input = tf.keras.layers.Input(shape=(1,), name='name_length')

    name_len_embd = tf.keras.layers.Embedding(NAME_LEN,5)(name_len_input)

    flatten7 = tf.keras.layers.Flatten()(name_len_embd)



    # concatenating the outputs

    concat_layer = tf.keras.layers.concatenate(inputs=[gru_name,gru_desc,flatten1,flatten2,flatten3,flatten4,input_shipping,flatten5,\

                                                       flatten6,flatten7],name="concatenate")

    #dense layers

    Dense_layer1 = tf.keras.layers.Dense(units=512,activation='relu',kernel_initializer='he_normal',\

                                         name="Dense_l")(concat_layer)

    dropout_1 = tf.keras.layers.Dropout(drop_out_list[0],name='dropout_1')(Dense_layer1)

    batch_n1 = tf.keras.layers.BatchNormalization()(dropout_1)

    

    Dense_layer2 = tf.keras.layers.Dense(units=256,activation='relu',kernel_initializer='he_normal',\

                                         name="Dense_2")(batch_n1)

    dropout_2 = tf.keras.layers.Dropout(drop_out_list[1],name='dropout_2')(Dense_layer2)

    batch_n2 = tf.keras.layers.BatchNormalization()(dropout_2)



    Dense_layer3 = tf.keras.layers.Dense(units=128,activation='relu',kernel_initializer='he_normal',\

                                         name="Dense_3")(batch_n2)

    dropout_3 = tf.keras.layers.Dropout(drop_out_list[2],name='dropout_3')(Dense_layer3)



    Dense_layer4 = tf.keras.layers.Dense(units=64,activation='relu',kernel_initializer='he_normal',\

                                         name="Dense_4")(dropout_3)

    dropout_4 = tf.keras.layers.Dropout(drop_out_list[3],name='dropout_4')(Dense_layer4)

    

    #output_layer

    final_output = tf.keras.layers.Dense(units=1,activation='linear',name='output_layer')(dropout_4)



    model = tf.keras.Model(inputs=[input_name,input_desc,input_brand,input_cat,input_subcat1,input_subcat2,\

                                   input_shipping,input_item,desc_len_input,name_len_input],

                           outputs=[final_output])

    # we specified the model input and output

    print(model.summary())

#    img_path = "GRU_model_2_lr.png"

#    plot_model(model, to_file=img_path, show_shapes=True, show_layer_names=True) 

    return model



#https://www.tensorflow.org/guide/keras/train_and_evaluate

def compile_predict_GRU(train_input,test_input,y_train,variable_list,drop_list):

    filepath="GRU_lr-{epoch:03d}-{val_loss:.3f}.hdf5"

    model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', mode='min', save_best_only=True)

    #tensorboard = tf.keras.callbacks.TensorBoard(log_dir='/kaggle/working/',\

     #                                        write_graph=True)



    #call_backs = [model_checkpoint,tensorboard]

    #https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler

    def scheduler(epoch):

        if epoch < 2:

            return 0.005

        else:

            return 0.001



    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

    call_backs = [model_checkpoint,lr_schedule]

    #call_backs = [model_checkpoint]

    #https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/

    x_train = train_input

    model= construct_GRU(x_train,variable_list,drop_list)

    model.compile(optimizer='adam', loss='mean_squared_error')

    # for i in range(3):

    #     with timer(f'epoch {i + 1}'):

    #         model_history= model.fit(x_train, y_train,\

    #                                            batch_size=2**(8 + i),\

    #                                            epochs=1, verbose=1,\

    #                                            callbacks=call_backs,\

    #                                            validation_split = 0.1)

    

    model.fit(x_train, y_train, epochs=3,batch_size=2**10, callbacks=call_backs,validation_split=0.1)



    #pred_on_train = model.predict(x_train,batch_size = 2**10,verbose = 1)

    pred_on_test = model.predict(test_input,batch_size = 2**10,verbose = 1)

    return pred_on_test



def get_tensorboard_ready():

    ! rm -rf ./logs/ 

    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    #logdir = ""

    print(logdir)

    return logdir



def hyperparameter_tuning_random(x,y,model_estimator,param_dict,cv_no):

    start = time.time()

    hyper_tuned = GridSearchCV(estimator = model_estimator, param_grid = param_dict,\

                                    return_train_score=True, scoring = 'neg_mean_squared_error',\

                                    cv = cv_no, \

                                    verbose=2, n_jobs = -1)

    hyper_tuned.fit(x,y)

    print("\n######################################################################\n")

    print ('Time taken for hyperparameter tuning is {} sec\n'.format(time.time()-start))

    print('The best parameters_: {}'.format(hyper_tuned.best_params_))

    return hyper_tuned.best_params_



def rmsle_compute(y_true, y_pred):

    assert len(y_true) == len(y_pred)

    score = np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))

    return score



def scale_back(x):

    '''

    Function to inverse transform the scaled values

    '''

    x= np.expm1(y_scalar.inverse_transform(x.reshape(-1,1))[:,0])#changes

    return x



@contextmanager

def timer(name):

    t0 = time.time()

    yield

    print(f'[{name}] done in {time.time() - t0:.0f} s')

def function_1(x):

    ############################

    #step1: load the train file

    ############################

    gc.collect()

    train = pd.read_csv('/kaggle/working/train/train.tsv',sep='\t')

    test = x

    print("Finished loading the files....\n")

    print("train: {0}\ntest: {1}\n".format(train.shape,test.shape))

    ##########################################################

    #step2: Data cleaning and preprocessing of train and test

    ##########################################################

    #https://www.kaggle.com/valkling/mercari-rnn-2ridge-models-with-notes-0-42755

    train = train.drop(train[(train.price < 3.0)].index)

    fill_nan(train)

    fill_nan(test)

    print("filled nan\n")



    train['wc_desc'] = train['item_description'].apply(lambda x: countwords(x))

    test['wc_desc'] = test['item_description'].apply(lambda x: countwords(x))

    train['wc_name'] = train['name'].apply(lambda x: countwords(x))

    test['wc_name'] = test['name'].apply(lambda x: countwords(x))

    create_split_categories(train)

    create_split_categories(test)

    print("Completed data cleaning and preprocessing\n")



    # train test split

    #X_train, X_test= train_test_split(train, train_size=0.99, random_state=123)

    print("shape of train: {}".format(train.shape))

    print("shape of test: {}".format(test.shape))



    global y_scalar,DESC_LEN,NAME_LEN

    y_scalar = StandardScaler()#changes

    y_train = y_scalar.fit_transform(log_price(train['price']).values.reshape(-1, 1))#changes

    #y_test = y_scalar.transform(log_price(X_test['price']).values.reshape(-1, 1))#



    DESC_LEN = train.wc_desc.max() + 1

    NAME_LEN = train.wc_name.max() + 1

    #################################

    #step:3 Featurizing

    #################################

    train_inputs,test_inputs,list_var = data_gru(train,test)



    #################################

    #step:4 compiling and predicting

    #################################

    dropout_list = [0.10,0.10,0.20,0.20]

    te_preds_m1 = compile_predict_GRU(train_inputs,test_inputs,y_train,list_var,dropout_list)



    dropout_list = [0.10,0.20,0.30,0.40]

    te_preds_m2 = compile_predict_GRU(train_inputs,test_inputs,y_train,list_var,dropout_list)



    del train_inputs,test_inputs,list_var

    gc.collect()

    

    #https://machinelearningmastery.com/model-averaging-ensemble-for-deep-learning-neural-networks/



    y_hats = np.array([te_preds_m1,te_preds_m2]) #making an array out of all the predictions

    # mean across ensembles

    mean_preds = np.mean(y_hats, axis=0)

    return scale_back(mean_preds)
#call for function_1

test = pd.read_csv('/kaggle/working/test/test_stg2.tsv',sep='\t')



test["price"] = function_1(test)

sub = test[["test_id", "price"]]

sub.to_csv("submission.csv", index = False)