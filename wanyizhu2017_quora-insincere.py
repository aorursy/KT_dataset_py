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

#stop_set = set(stopwords.words('english'))

#lemmatizer = WordNetLemmatizer()



# with lemmatization

#train['tokens'] = train['question_text'].map(lambda x: tokenize(x, stop_set, lemmatizer))



# without lemmatization

train['tokens'] = train['question_text'].map(lambda x: tokenize(x))

def build_vocab(token_col):

    

    vocab = {}

    for tokens in token_col:

        for token in tokens:

            vocab[token] = vocab.get(token, 0) + 1



    return vocab



train_vocab = build_vocab(train['tokens'])
from gensim.models import KeyedVectors



news_path = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'

embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)
import operator



def check_coverage(vocab,embedding):

    

    oov = {}

    k = 0

    i = 0

    

    for word in vocab:

        if word in embedding:

            k += vocab[word]

        else:

            oov[word] = vocab[word]

            i += vocab[word]



    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))

    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]



    return sorted_x

  



not_found_vocab = check_coverage(train_vocab, embeddings_index)
#'Aadhar' in embeddings_index
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
train_vocab = build_vocab(train['clean_tokens'])

not_found_vocab = check_coverage(train_vocab, embeddings_index)

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
# free up RAM

import gc



del not_found_vocab

del embeddings_index

del train_vocab

del train



gc.collect()
from sklearn import linear_model, tree, ensemble, metrics, model_selection, exceptions





def print_score(y_true, y_pred):

    print(' accuracy : ', metrics.accuracy_score(y_true, y_pred))

    print('precision : ', metrics.precision_score(y_true, y_pred))

    print('   recall : ', metrics.recall_score(y_true, y_pred))

    print('       F1 : ', metrics.f1_score(y_true, y_pred))



    

# train-test split

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state = 2019)

#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, train_size = 0.2, random_state = 2019)

np.random.seed(2019)



# biased sampling

# def select_train(X, y):

#     pos_index = np.where(y == 1)[0]

#     neg_index = np.where(y == 0)[0]

#     size_select = min(len(pos_index), len(neg_index)) // 2

#     return np.sort(np.append(np.random.choice(pos_index, size_select, replace = False), np.random.choice(neg_index, size_select, replace = False)))



# train_index = select_train(X_train, y_train)

# val_index = np.setdiff1d(range(len(X_train)), train_index)

# X_trt, y_trt, X_trv, y_trv = [0, 0], [0, 0], [0, 0], [0, 0]

# X_trt[1], y_trt[1] = X_train[train_index,:], y_train[train_index]

# X_trv[1], y_trv[1] = X_train[val_index,:], y_train[val_index]



# X_trt[0], X_trv[0], y_trt[0], y_trv[0] = model_selection.train_test_split(X_train, y_train, test_size = len(X_trv[1]), random_state = 2019)
# free up RAM

import gc



#del not_found_vocab

#del embeddings_index

#del train_vocab

#del train



del X

del y

gc.collect()
# use full training dataset with cross validation

lr = linear_model.LogisticRegression(solver = 'liblinear')



cv_score = model_selection.cross_val_score(lr, X_train, y_train, cv = 5)

print("Cross validation score:")

print(cv_score)



 

lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

y_pred_proba = lr.predict_proba(X_test)[:,1]



print_score(y_test, y_pred_lr)
# threshold search



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
# # With biased sampling



# lr = [linear_model.LogisticRegression(solver = 'liblinear') for _ in range(2)]



# y_val_lr, y_pred_lr = [0, 0], [0, 0]

# for i in range(2):

#     lr[i].fit(X_trt[i], y_trt[i])

#     y_val_lr[i] = lr[i].predict(X_trv[i])

#     y_pred_lr[i] = lr[i].predict(X_test)

    

# print('-- validation result comparison --')

# for i in range(2):

#     print('- with' + ('' if i else 'out') + ' biased sampling -')

#     print_score(y_trv[i], y_val_lr[i])

# print('-- test result comparison --')

# for i in range(2):

#     print('- with' + ('' if i else 'out') + ' biased sampling -')

#     print_score(y_test, y_pred_lr[i])    
# probably need more thoughts on if to use Gaussian or to use multinominal with TF-IDF



from sklearn import naive_bayes



nb = naive_bayes.GaussianNB()



cv_score = model_selection.cross_val_score(nb, X_train, y_train, cv = 5)

print("Cross validation score:")

print(cv_score)





nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)

y_pred_proba = nb.predict_proba(X_test)[:,1]

print_score(y_test, y_pred_nb)
# threshold search



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
rf = ensemble.RandomForestClassifier(n_estimators = 80, random_state = 2019, max_depth = 100)



cv_score = model_selection.cross_val_score(rf, X_train, y_train, cv = 5)

print("Cross validation score:")

print(cv_score)





rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

y_pred_proba = rf.predict_proba(X_test)[:,1]

print_score(y_test, y_pred_rf)
# threshold search



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
# rf = [ensemble.RandomForestClassifier(n_estimators = 500, random_state = 2019, max_depth = 200) for _ in range(2)]



# y_val_rf, y_pred_rf = [0, 0], [0, 0]

# for i in range(2):

#     rf[i].fit(X_trt[i], y_trt[i])

#     y_val_rf[i] = rf[i].predict(X_trv[i])

#     y_pred_rf[i] = rf[i].predict(X_test)

    

    

# print('-- validation result comparison --')

# for i in range(2):

#     print('- with' + ('' if i else 'out') + ' biased sampling -')

#     print_score(y_trv[i], y_val_rf[i])

# print('-- test result comparison --')

# for i in range(2):

#     print('- with' + ('' if i else 'out') + ' biased sampling -')

#     print_score(y_test, y_pred_rf[i])    
# import lightgbm as lgb



# lgb_c = lgb.LGBMClassifier(learning_rate = 0.04,n_estimators = 3000)



# lgb_c.fit(X_train, y_train,

#           eval_set = [(X_test, y_test)],

#           early_stopping_rounds = 5,

#           eval_metric = 'auc',

#           verbose = 50)





# y_pred = lgb_c.predict(X_test, num_iteration=lgb_c.best_iteration_)

# print_score(y_test, y_pred)
# # threshold search



# y_pred_proba = lgb_c.predict_proba(X_test,num_iteration=lgb_c.best_iteration_)[:,1]



# thresholds = []

# for thresh in np.arange(0.1, 0.91, 0.01):

#     thresh = np.round(thresh, 2)

#     res = metrics.f1_score(y_test, (y_pred_proba > thresh).astype(int))

#     thresholds.append([thresh, res])

#     # print("F1 score at threshold {0} is {1}".format(thresh, res))



# thresholds.sort(key=lambda x: x[1], reverse=True)

# best_thresh = thresholds[0][0]

# print("Best threshold: ", best_thresh)

# print("Best F1: ", thresholds[0][1])
# import lightgbm as lgb



# lgb_c = lgb.LGBMClassifier(learning_rate = 0.04,n_estimators = 3000, boosting_type = 'dart', scale_pos_weight = 3.5)



# lgb_c.fit(X_train, y_train,

#           eval_set = [(X_test, y_test)],

#           early_stopping_rounds = 5,

#           eval_metric = 'auc',

#           verbose = 50)





# y_pred = lgb_c.predict(X_test, num_iteration=lgb_c.best_iteration_)

# print_score(y_test, y_pred)
# import lightgbm as lgb



# lgb_c = lgb.LGBMClassifier(learning_rate = 0.04,n_estimators = 2000)



# lgb_c.fit(X_train, y_train,

#           eval_set = [(X_test, y_test)],

#           early_stopping_rounds = 5,

#           eval_metric = 'auc',

#           verbose = 50)





# y_pred = lgb_c.predict(X_test, num_iteration=lgb_c.best_iteration_)

# print_score(y_test, y_pred)
# import lightgbm as lgb



# lgb_c = lgb.LGBMClassifier(learning_rate = 0.04,n_estimators = 2000, scale_pos_weight = 3.5)



# lgb_c.fit(X_train, y_train,

#           eval_set = [(X_test, y_test)],

#           early_stopping_rounds = 5,

#           eval_metric = 'auc',

#           verbose = 50)





# y_pred = lgb_c.predict(X_test, num_iteration=lgb_c.best_iteration_)

# print_score(y_test, y_pred)
# import lightgbm as lgb



# gridParams = {

#     'num_leaves':[20, 60],

#     'max_depth': [-1, 50, 200],

#     'reg_alpha' : [0, 1,1.2],

#     }



# lgb_c = lgb.LGBMClassifier(n_estimators = 3000, boosting_type = 'dart')

# grid_lgb = model_selection.RandomizedSearchCV(lgb_c, gridParams, scoring='accuracy', cv = 3)





# grid_lgb.fit(X_train, y_train)



# print(grid_lgb.best_params_)

# print(grid_lgb.best_score_)
# import tensorflow as tf



# model = tf.keras.models.Sequential([

#     tf.keras.layers.Dense(12, activation=tf.nn.relu, input_shape=(300,)),

#     tf.keras.layers.Dense(100, activation=tf.nn.relu),

#     #tf.keras.layers.Dense(100, activation=tf.nn.relu),

#     tf.keras.layers.Dropout(0.2),

#     tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

# ])



# model.compile(optimizer=tf.train.AdamOptimizer(0.001),

#               loss='binary_crossentropy',

#               metrics=['accuracy'])



# model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# y_pred = model.predict(X_test)
# print_score(y_test, np.round(y_pred))
# test = pd.read_csv('../input/test.csv')

# test['tokens'] = test['question_text'].map(lambda x: tokenize(x))