import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
import nltk

#nltk.download('popular')



import re

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

from nltk.tokenize import TweetTokenizer



# text cleaning & tokenization

def tokenize(text, stop_set = None, lemmatizer = None):

    

    # clean text

    text = text.encode('ascii', 'ignore').decode('ascii')

    #text = text.lower()

    

    text = re.sub(r'\b(?:(?:https?|ftp)://)?\w[\w-]*(?:\.[\w-]+)+\S*', ' ', text) # remove hyperlink,subs charact in the brackets

    text = re.sub("[\r\n]", ' ', text) # remove new line characters

    #text = re.sub(r'[^\w\s]','',text)

    text = text.strip()

    

    #tokens = word_tokenize(text)

    # use TweetTokenizer instead of word_tokenize -> to prevent splitting at apostrophies

    tknzr = TweetTokenizer()

    tokens = tknzr.tokenize(text)

    

    # retain tokens with at least two words

    tokens = [token for token in tokens if re.match(r'.*[a-z]{1,}.*', token)]

    

    # remove stopwords - optional

    # removing stopwords lost important information

    if stop_set != None:

        tokens = [token for token in tokens if token not in stop_set]

    

    # lemmmatization - optional

    if lemmatizer != None:

        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens





train['tokens'] = train['question_text'].map(lambda x: tokenize(x))
from gensim.models import KeyedVectors



news_path = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'

embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)
# Google News Embeddings

# replace not found words

to_remove = ['to','of','and', 'a']



#replace_dict = {}

#replace_dict = {'quora':'Quora', 'i\'ve':'I\'ve', 'instagram':'Instagram', 'upsc':'UPSC', 'bitcoin':'Bitcoin', 'trump\'s':'Trump',

#               'mbbs':'MBBS', 'whatsapp':'WhatsApp', 'favourite':'favorite', 'ece':'ECE', 'aiims':'AIIMS', 'colour':'color',

#               'doesnt':'doesn\'t','centre':'center','sbi':'SBI','cgl':'CGL','iim':'IIM','btech':'BTech'}



replace_dict = {'favourite':'favorite', 'bitcoin':'Bitcoin', 'colour':'color', 'doesnt':'doesn\'t', 'centre':'center', 'Quorans':'Quora',

               'travelling':'traveling', 'counselling':'counseling', 'didnt':'didn\'t', 'btech':'BTech','isnt':'isn\'t',

               'Shouldn\'t':'shouldn\'t', 'programme':'program', 'realise':'realize', 'Wouldn\'t':'wouldn\'t', 'defence':'defense',

               'Aren\'t':'aren\'t', 'organisation':'organization', 'How\'s':'how\'s', 'e-commerce':'ecommerce', 'grey':'gray',

               'bitcoins':'Bitcoin', 'honours':'honors', 'learnt':'learned', 'licence':'license', 'mtech':'MTech', 'colours':'colors',

               'e-mail':'email', 't-shirt':'tshirt', 'Whatis':'What\'s', 'theatre':'theater', 'labour':'labor', 'Isnt':'Isn\'t',

               'behaviour':'behavior','aadhar':'Aadhar', 'Qoura':'Quora', 'aluminium':'aluminum'}



def clean_token(tokens, remove_list, re_dict, embedding):

    

    c_tokens = []

    for token in tokens:

        if token not in remove_list:

            token2 = token

            if token2 in embedding:

                c_tokens.append(token2)

            elif token2 in re_dict:

                token2 = re_dict[token2]

                c_tokens.append(token2)

            else:    

                # apostrophe

                if token2.endswith('\'s'):

                    token2 = token2[:-2]

                    

                if (token2.endswith('s')) & (token2[:-1] in embedding):

                    token2 = token2[:-1]

                    

                # break dash

                if "-" in token2:

                    token2 = token2.split('-')

                    c_tokens += token2

                else:

                    c_tokens.append(token2)

        



    return c_tokens



train['clean_tokens'] = train['tokens'].map(lambda x: clean_token(x, to_remove, replace_dict, embeddings_index))
def doc_mean(tokens, embedding):

    

    e_values = []

    e_values = [embedding[token] for token in tokens if token in embedding]

    

    if len(e_values) > 0:

        return np.mean(np.array(e_values), axis=0)

        #return np.sum(np.array(e_values), axis=0)

    else:

        #return np.ones(300)*-999

        return np.zeros(300)

      

X = np.vstack(train['clean_tokens'].apply(lambda x: doc_mean(x, embeddings_index)))

#X = np.vstack(train['tokens'].apply(lambda x: doc_mean(x, embeddings_index)))



y = train['target'].values

indices = train.index
# free up RAM

import gc



del embeddings_index

#del train



gc.collect()
from sklearn import linear_model, tree, ensemble, metrics, model_selection, exceptions





def print_score(y_true, y_pred):

    print(' accuracy : ', metrics.accuracy_score(y_true, y_pred))

    print('precision : ', metrics.precision_score(y_true, y_pred))

    print('   recall : ', metrics.recall_score(y_true, y_pred))

    print('       F1 : ', metrics.f1_score(y_true, y_pred))



    

# train-test split

X_train, X_test, y_train, y_test, train_indices, test_indices = model_selection.train_test_split(X, y, indices, test_size = 0.2, random_state = 2019)



# train-test split - small sample

#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, train_size = 0.2, random_state = 2019)
# free up RAM

import gc



#del X

del y

#del train



gc.collect()
# import lightgbm as lgb



# lgb_c = lgb.LGBMClassifier(learning_rate = 0.02,n_estimators = 2000)



# lgb_c.fit(X_train, y_train,

#           eval_set = [(X_test, y_test)],

#           early_stopping_rounds = 5,

#           eval_metric = 'auc',

#           verbose = 100)





# y_pred = lgb_c.predict(X_test, num_iteration=lgb_c.best_iteration_)

# print_score(y_test, y_pred)
# treshold search



# y_pred_proba = lgb_c.predict_proba(X_test,num_iteration=lgb_c.best_iteration_)[:,1]



# thresholds = []

# for thresh in np.arange(0.1, 0.901, 0.01):

#     res = metrics.f1_score(y_test, (y_pred_proba > thresh).astype(int))

#     thresholds.append([thresh, res])

#     # print("F1 score at threshold {0} is {1}".format(thresh, res))



# thresholds.sort(key=lambda x: x[1], reverse=True)

# best_thresh = thresholds[0][0]

# print("Best threshold: ", best_thresh)

# print("Best F1: ", thresholds[0][1])
# import lightgbm as lgb



# gridParams = {

#     'scale_pos_weight': [3, 3.5, 4, 4.5]

#     }



# lgb_c = lgb.LGBMClassifier(learning_rate = 0.02,n_estimators = 2000)

# grid_lgb = model_selection.GridSearchCV(lgb_c, gridParams, scoring='f1', cv = 3)





# grid_lgb.fit(X_train, y_train, 

#           eval_set = [(X_test, y_test)],

#           early_stopping_rounds = 5,

#           eval_metric = 'auc',

#           verbose = 500)



# print(grid_lgb.best_params_)

# print(grid_lgb.best_score_)
import lightgbm as lgb



lgb_c = lgb.LGBMClassifier(learning_rate = 0.04,n_estimators = 3200, boosting_type = 'dart')



lgb_c.fit(X_train, y_train,

          eval_set = [(X_test, y_test)],

          eval_metric = 'auc',

          verbose = 500)





y_pred = lgb_c.predict(X_test, num_iteration=lgb_c.best_iteration_)

print_score(y_test, y_pred)
# treshold search



y_pred_proba = lgb_c.predict_proba(X_test,num_iteration=lgb_c.best_iteration_)[:,1]



thresholds = []

for thresh in np.arange(0.1, 0.91, 0.01):

    thresh = np.round(thresh, 2)

    res = metrics.f1_score(y_test, (y_pred_proba > thresh).astype(int))

    thresholds.append([thresh, res])

    # print("F1 score at threshold {0} is {1}".format(thresh, res))



thresholds.sort(key=lambda x: x[1], reverse=True)

best_thresh = thresholds[0][0]

print("Best threshold: ", best_thresh)

print("Best F1: ", thresholds[0][1])
import lightgbm as lgb



lgb_c_weight = lgb.LGBMClassifier(learning_rate = 0.04,n_estimators = 3200, boosting_type = 'dart', scale_pos_weight = 3.5)



lgb_c_weight.fit(X_train, y_train,

          eval_set = [(X_test, y_test)],

          eval_metric = 'auc',

          verbose = 500)





y_pred_weight = lgb_c_weight.predict(X_test, num_iteration=lgb_c_weight.best_iteration_)

print_score(y_test, y_pred_weight)
p_labels = lgb_c_weight.predict(X, num_iteration=lgb_c_weight.best_iteration_)

p_proba = lgb_c_weight.predict_proba(X, num_iteration=lgb_c_weight.best_iteration_)



output_np = np.concatenate((p_labels.reshape(len(p_labels), 1), p_proba), axis = 1)

output = pd.DataFrame(output_np)

output.to_csv('label with proba.csv')