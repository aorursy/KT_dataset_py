import time
import random
import pandas as pd
import numpy as np
import gc
import re
import torch
from torchtext import data
import spacy
from tqdm import tqdm_notebook, tnrange
from tqdm.auto import tqdm

tqdm.pandas(desc='Progress')
from collections import Counter
from textblob import TextBlob
from nltk import word_tokenize

import lightgbm as lgb
import xgboost as xgb

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torchtext.data import Example
from sklearn.metrics import f1_score
import torchtext
import os 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# cross validation and metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
# from torch.optim.optimizer import Optimizer
from unidecode import unidecode
embed_size = 400 # how big is each word vector
max_features = 120000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70 # max number of words in a question to use
batch_size = 512 # how many samples to process at once
n_epochs = 5 # how many times to iterate over all samples
n_splits = 10 # Number of K-fold Splits

SEED = 229
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()
## FUNCTIONS TAKEN FROM https://www.kaggle.com/gmhost/gru-capsule
'''
def load_embeddings(word_index, EMBEDDING_FILE):
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))[:400]
    
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    print('File {}: all_embs.mean = {}, all_embs.std = {}'.format(EMBEDDING_FILE, emb_mean, emb_std))
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    # Why random embedding for OOV? what if use mean?
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    #embedding_matrix = np.random.normal(emb_mean, 0, (nb_words, embed_size)) # std 0
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            
    return embedding_matrix 


load_glove = lambda word_index: load_embeddings(word_index, EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
load_fasttext = lambda word_index: load_embeddings(word_index, EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec')
load_para = lambda word_index: load_embeddings(word_index, EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt')
'''
def load_glove(word_index):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if o.split(" ")[0] in word_index)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            
    return embedding_matrix 
    
def load_fasttext(word_index):    
    EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100 and o.split(" ")[0] in word_index )

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix

def load_para(word_index):
    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100 and o.split(" ")[0] in word_index)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    
    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    return embedding_matrix
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df = pd.concat([df_train ,df_test],sort=True)
q = df["question_text"].fillna("na_").values

char_vector = TfidfVectorizer(
    ngram_range=(2,4),
    max_features=20000,
    stop_words='english',
    analyzer='char_wb',
    token_pattern=r'\w{1,}',
    strip_accents='unicode',
    sublinear_tf=True, 
    max_df=0.98,
    min_df=2
)
char_vector.fit(q[:85000])

char_vector = char_vector.transform(q).tocsr()
word_vector = TfidfVectorizer(
    ngram_range=(1,1), 
    max_features=9000,
    sublinear_tf=True, 
    strip_accents='unicode', 
    analyzer='word', 
    token_pattern="\w{1,}", 
    stop_words="english",
    max_df=0.95,
    min_df=2
)
word_vector.fit(q)

word_vector = word_vector.transform(q).tocsr()
del q
gc.collect()
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
list(vocab.keys())[:10]
sin = len(df_train[df_train["target"]==0])
insin = len(df_train[df_train["target"]==1])
persin = (sin/(sin+insin))*100
perinsin = (insin/(sin+insin))*100            
print("# Sincere questions: {:,}({:.2f}%) and # Insincere questions: {:,}({:.2f}%)".format(sin,persin,insin,perinsin))
# print("Sinsere:{}% Insincere: {}%".format(round(persin,2),round(perinsin,2)))
print("# Test samples: {:,}({:.3f}% of train samples)".format(len(df_test),len(df_test)/len(df_train)))
def get_wired(d):
    wired = []
    for v in d:
        m = re.match('[\w{}]+', v)
        if not m or m[0] != v:
            wired += [v]
    return wired

wired = get_wired(vocab)
len(wired)
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
from sklearn.preprocessing import StandardScaler

def known_contractions(embed):
    known = []
    for contract in contraction_mapping:
        if contract in embed:
            known.append(contract)
    return known


def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text


def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x


def unknown_punct(embed, punct):
    unknown = ''
    for p in punct:
        if p not in embed:
            unknown += p
            unknown += ' '
    return unknown


def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x


def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text


def add_lower(embedding, vocab):
    count = 0
    for word in vocab:
        if word in embedding and word.lower() not in embedding:  
            embedding[word.lower()] = embedding[word]
            count += 1
    print(f"Added {count} words to embedding")


def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x


def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x


def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispellings, mispellings_re = _get_mispell(mispell_dict)
def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)


# Extra feature part taken from https://github.com/wongchunghang/toxic-comment-challenge-lstm/blob/master/toxic_comment_9872_model.ipynb
def add_features(df):
    df['question_text'] = df['question_text'].progress_apply(lambda x:str(x))
    df['total_length'] = df['question_text'].progress_apply(len)
    df['capitals'] = df['question_text'].progress_apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.progress_apply(lambda row: float(row['capitals'])/float(row['total_length']), axis=1)
    df['num_words'] = df.question_text.str.count('\S+')
    df['num_unique_words'] = df['question_text'].progress_apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
    df['char_vector'] = char_vector
    df['word_vector'] = word_vector
    
    df['question_text'] = df['question_text'].progress_apply(lambda x: x.lower())
    df['question_text'] = df['question_text'].progress_apply(clean_text)
    df['question_text'] = df['question_text'].progress_apply(clean_numbers)
    df['question_text'] = df['question_text'].progress_apply(replace_typical_misspell)
    return df
 ## fill up the missing values
X = df["question_text"].fillna("_##_").values

###################### Add Features ###############################
#  https://github.com/wongchunghang/toxic-comment-challenge-lstm/blob/master/toxic_comment_9872_model.ipynb
df = add_features(df)

features = df[['caps_vs_length', 'words_vs_unique']].fillna(0)

ss = StandardScaler()
ss.fit(features)
features = ss.transform(features)
###########################################################################

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X))
X = tokenizer.texts_to_sequences(X)

## Pad the sentences 
X = pad_sequences(X, maxlen=maxlen)
X_train, X_test = X[:len(df_train)], X[len(df_train):]
features_train, features_test = features[:len(df_train)], features[len(df_train):]
df_train, df_test = df.iloc[:len(df_train)], df.iloc[len(df_train):]

## Get the target values
y_train = df_train['target'].values

#     # Splitting to training and a final test set    
#     train_X, x_test_f, train_y, y_test_f = train_test_split(list(zip(train_X,features)), train_y, test_size=0.2, random_state=SEED)    
#     train_X, features = zip(*train_X)
#     x_test_f, features_t = zip(*x_test_f)    

#shuffling the data
np.random.seed(SEED)
sample = np.random.permutation(len(X_train))

X_train = X_train[sample]
y_train = y_train[sample]
features_train = features_train[sample]

#     return train_X, test_X, train_y, features, test_features, tokenizer.word_index
#     return train_X, test_X, train_y, x_test_f,y_test_f,features, test_features, features_t, tokenizer.word_index
#     return train_X, test_X, train_y, tokenizer.word_index
np.save("x_train", X_train)
np.save("x_test", X_test)
np.save("y_train", y_train)

np.save("features", features_train)
np.save("test_features",features_test)
np.save("word_index.npy",tokenizer.word_index)
del X_train
del X_test
del y_train
del features_train
del features_test
del tokenizer.word_index
gc.collect()
X_train = np.load("x_train.npy")
X_test = np.load("x_test.npy")
y_train = np.load("y_train.npy")
features_train = np.load("features.npy")
features_test = np.load("test_features.npy")
word_index = np.load("word_index.npy").item()
input_train, input_valid, y_train, y_val = train_test_split(X_train, y_train, train_size=.9, random_state=SEED)
features.shape
# missing entries in the embedding are set using np.random.normal so we have to seed here too
seed_everything()

glove_embeddings = load_glove(word_index)
paragram_embeddings = load_para(word_index)
fasttext_embeddings = load_fasttext(word_index)

embedding_matrix = np.mean([glove_embeddings, paragram_embeddings, fasttext_embeddings], axis=0)

# vocab = build_vocab(df['question_text'])
# add_lower(embedding_matrix, vocab)
del glove_embeddings, paragram_embeddings, fasttext_embeddings
gc.collect()

np.shape(embedding_matrix)
# splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(X_train, y_train))
# splits[:3]
'''
def train_model(x, y, lgb_params, 
                number_of_folds=10, 
                evaluation_metric='auc', 
                save_feature_importances=False, 
                early_stopping_rounds=50, 
                num_round = 50,
                identifier_columns=['MachineIdentifier'],
                single_fold=False):
    cross_validator = StratifiedKFold(n_splits=number_of_folds,
                                  random_state=random_state,
                                  shuffle=shuffle)
    
    validation_scores = []
    classifier_models = []
    feature_importance_df = pd.DataFrame()
    for fold_index, (train_index, validation_index) in enumerate(cross_validator.split(x, y)):
        x_train, x_validation = x.iloc[train_index], x.iloc[validation_index]
        y_train, y_validation = y.iloc[train_index], y.iloc[validation_index]
    
        x_train.drop(identifier_columns, axis=1, inplace=True)
        validation_identifier_data = x_validation[identifier_columns]
        x_validation.drop(identifier_columns, axis=1, inplace=True)
        x_train_columns = x_train.columns
        trn_data = lgb.Dataset(x_train,
                       label=y_train,
                       # categorical_feature=categorical_columns
                       )
        del x_train
        del y_train
        val_data = lgb.Dataset(x_validation,
                               label=y_validation,
                               # categorical_feature=categorical_columns
                               )
        classifier_model = lgb.train(lgb_params,
                                     trn_data,
                                      num_round,
                                     valid_sets=[trn_data, val_data],
                                     verbose_eval=100,
                                     early_stopping_rounds=early_stopping_rounds
                                     )

        classifier_models.append(classifier_model)
        
        predictions = classifier_model.predict(x_validation, num_iteration=classifier_model.best_iteration)
        false_positive_rate, recall, thresholds = metrics.roc_curve(y_validation, predictions)
        score = metrics.auc(false_positive_rate, recall)
        validation_scores.append(score)
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = x_train_columns
        fold_importance_df["importance"] = classifier_model.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = fold_index + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        if single_fold:
            break
    if save_feature_importances:
        cols = (feature_importance_df[["feature", "importance"]]
                .groupby("feature")
                .mean()
                .sort_values(by="importance", ascending=False)[:1000].index)

        best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

        plt.figure(figsize=(14, 25))
        sns.barplot(x="importance",
                    y="feature",
                    data=best_features.sort_values(by="importance",
                                                   ascending=False))
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        plt.savefig('lgbm_importances.png')

        # mean_gain = feature_importances[['gain', 'feature']].groupby('feature').mean()
        # feature_importances['mean_gain'] = feature_importances['feature'].map(mean_gain['gain'])
        #
        # temp = feature_importances.sort_values('mean_gain', ascending=False)
        best_features.sort_values(by="importance", ascending=False) \
            .groupby("feature") \
            .mean() \
            .sort_values(by="importance", ascending=False) \
            .to_csv('feature_importances_new.csv', index=True)

    score = sum(validation_scores) / len(validation_scores)
    return classifier_models, score
'''
'''reference: some settings inspired by Toxic competition kernels'''
def build_xgb(train_X, train_y, valid_X, valid_y=None, subsample=0.75):

    xgtrain = xgb.DMatrix(train_X, label=train_y)
    if valid_y is not None:
        xgvalid = xgb.DMatrix(valid_X, label=valid_y)
    else:
        xgvalid = None
    
    model_params = {}
    # binary 0 or 1
    model_params['objective'] = 'binary:logistic'
    # eta is the learning_rate, [default=0.3]
    model_params['eta'] = 0.3
    # depth of the tree, deeper more complex.
    model_params['max_depth'] = 3
    # 0 [default] print running messages, 1 means silent mode
    model_params['silent'] = 0
    model_params['eval_metric'] = 'auc'
    # will give up further partitioning [default=1]
    model_params['min_child_weight'] = 1
    # subsample ratio for the training instance
    model_params['subsample'] = subsample
    # subsample ratio of columns when constructing each tree
    model_params['colsample_bytree'] = subsample
    # random seed
    model_params['seed'] = SEED
    # imbalance data ratio
    #model_params['scale_pos_weight'] = 
    
    # convert params to list
    model_params = list(model_params.items())
    
    return xgtrain, xgvalid, model_params
def train_xgboost(xgtrain, xgvalid, model_params, num_rounds=500, patience=20):
    
    if xgvalid is not None:
        # watchlist what information should be printed. specify validation monitoring
        watchlist = [ (xgtrain, 'train'), (xgvalid, 'test') ]
        #early_stopping_rounds = stop if performance does not improve for k rounds
        model = xgb.train(model_params, xgtrain, num_rounds, watchlist, early_stopping_rounds=patience)
    else:
        model = xgb.train(model_params, xgtrain, num_rounds)
    
    return model
# # params from https://www.kaggle.com/fabiendaniel/detecting-malwares-with-lgbm
# params = {'num_leaves': 128,
#          'min_data_in_leaf': 42,
#          'objective': 'binary',
#          'max_depth': -1,
#          'learning_rate': 0.05,
#          "boosting": "gbdt",
#          "feature_fraction": 0.8,
#          "bagging_freq": 5,
#          "bagging_fraction": 0.8,
#          "bagging_seed": 11,
#          "lambda_l1": 0.15,
#          "lambda_l2": 0.15,
#          "random_state": 42,          
#          "verbosity": -1}

base_params = {   
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'nthread': 4,
        'learning_rate': 0.05,
        'max_depth': 3,
        'num_leaves': 40,
        'sub_feature': 0.9,
        'sub_row':0.9,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'random_state': SEED
        }
%%time
# models, validation_score = train_model(train.drop('HasDetections', axis=1),
#                                       train_y, base_params,
#                                       num_round=5000,
#                                       single_fold=stop_after_one_fold,
#                                       save_feature_importances=True)
print('train the model')
xgtrain, xgvalid, model_params = build_xgb(input_train, y_train ,input_valid, y_val)
model = train_xgboost(xgtrain, xgvalid, model_params)
def bestThresshold(y_train,train_preds):
    tmp = [0,0,0] # idx, cur, max
    delta = 0
    for tmp[0] in tqdm(np.arange(0.1, 0.501, 0.01)):
        tmp[1] = f1_score(y_train, np.array(train_preds)>tmp[0])
        if tmp[1] > tmp[2]:
            delta = tmp[0]
            tmp[2] = tmp[1]
    print('best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, tmp[2]))
    return delta

train_preds = np.zeros((input_train.shape[0], 1))
train_preds[:,0] = model.predict(xgb.DMatrix(input_train), ntree_limit=model.best_ntree_limit)
delta = bestThresshold(y_train,train_preds)
print('predict results')
test_preds = np.zeros(( X_test.shape[0], 1 ))
test_preds[:,0] = model.predict(xgb.DMatrix(X_test), ntree_limit=model.best_ntree_limit)
submission = df_test[['qid']].copy()
submission['prediction'] = (test_preds > delta).astype(int)
submission.to_csv('submission.csv', index=False)
!head submission.csv