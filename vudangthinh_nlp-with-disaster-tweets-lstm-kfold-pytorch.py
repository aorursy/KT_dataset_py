!pip install ekphrasis
import os

import re

import random

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.nn import Module

from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split



import string

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer

from ekphrasis.classes.preprocessor import TextPreProcessor

from ekphrasis.classes.tokenizer import SocialTokenizer

from ekphrasis.dicts.emoticons import emoticons

from keras.preprocessing.sequence import pad_sequences



from tqdm import tqdm, trange, tqdm_notebook

from sklearn.metrics import accuracy_score, f1_score

from sklearn.model_selection import StratifiedKFold



import gensim

from gensim.models import KeyedVectors



from IPython.core.debugger import set_trace
args = {

    'data_dir': '../input/nlp-getting-started/',

    'output_dir': '../working/',

    'train_file': 'train.csv',

    'test_file': 'test.csv',

    'w2v_pretrain': '../input/glove-twitter-pretraining-for-word-embedding/glove.twitter.27B.50d.txt',

    

    'hidden_dim': 256,

    'target_dim': 1,

    'max_len': 20,

    'batch_size': 32,

    'n_epoch': 10,

    'n_splits': 5,

    'learning_rate': 1e-3,

    'seed': 1234,

}
def search_threshold(y_true, preds):

    '''Find the best threshold value'''

    best_thresh = 0.5

    best_score = 0.0

    for thresh in np.arange(0.3, 0.501, 0.01):

        thresh = np.round(thresh, 2)

        score = f1_score(y_true, (preds > thresh).astype(int))

        if score > best_score:

            best_thresh = thresh

            best_score = score

    return best_thresh, best_score



def seed_everything(seed=1234):

    '''Setting seed values to get reproducible results'''

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

seed_everything(args['seed'])
# Load word embedding.

word_vectors = KeyedVectors.load_word2vec_format(args['w2v_pretrain'], binary=False)

embed_dim = word_vectors.vector_size
# Define pre-process functions for Tweet content.

def create_text_processor():

    text_processor = TextPreProcessor(

            normalize=['url', 'email', 'percent', 'money', 'phone', 'user',

                       'time', 'date', 'number'],

            fix_html=True,

            segmenter="twitter",

            corrector="twitter",



            unpack_hashtags=True,

            unpack_contractions=True,

            spell_correct_elong=True,



            # tokenizer=SocialTokenizer(lowercase=True).tokenize,

            tokenizer=RegexpTokenizer(r'\w+').tokenize,



            dicts=[emoticons]

        )



    return text_processor



def remove_stopword(tokens):

    stop_words = stopwords.words('english')

    return [word for word in tokens if word not in stop_words]



def stemming(tokens, ps):

    tokens = [ps.stem(token) for token in tokens]

    return tokens



def lemmatizer(tokens, wn):

    tokens = [wn.lemmatize(token) for token in tokens]

    return tokens

    

def pre_process(s):

    text = s.text

    text = text.replace("\n", " ")

    text = text.replace("\/", '/')

    text = text.lower()



    tokens = text_processor.pre_process_doc(text)

    tokens = remove_stopword(tokens)

    tokens = stemming(tokens, ps)

    tokens = lemmatizer(tokens, wn)

    return tokens



def token_to_index(s):

    '''Convert token to its index in the word embedding model'''

    tokens = s.tokens

    return [word_vectors.vocab.get(token).index if token in word_vectors else 0 for token in tokens]
# Load data, pre-process text and convert tokens to indices 

def load_data():

    train_df = pd.read_csv(os.path.join(args['data_dir'], args['train_file']))

    test_df = pd.read_csv(os.path.join(args['data_dir'], args['test_file']))

    

    train_df['tokens'] = train_df.apply(pre_process, axis=1)

    test_df['tokens'] = test_df.apply(pre_process, axis=1)

    train_df['token_index'] = train_df.apply(token_to_index, axis=1)

    test_df['token_index'] = test_df.apply(token_to_index, axis=1)

    

    train_X = pad_sequences(train_df['token_index'].values, maxlen=args['max_len'])

    test_X = pad_sequences(test_df['token_index'].values, maxlen=args['max_len'])

    train_y = train_df['target'].values

    

    return train_X, test_X, train_y
# Define training model

class Net(Module):

    def __init__(self, hidden_dim, target_size):

        super(Net, self).__init__()

        weights = torch.FloatTensor(word_vectors.vectors)

        oov_vector = torch.zeros([1, weights.shape[1]])

        weights = torch.cat((weights, oov_vector), 0)



        self.embedding = nn.Embedding.from_pretrained(weights)

        self.embedding.weight.requires_grad = False

        

        self.dropout_1 = nn.Dropout(p=0.2)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)

        self.dropout_2 = nn.Dropout(p=0.2)

        self.linear_out = nn.Linear(hidden_dim, target_size)

        

    def forward(self, content):

        embed_vector = self.embedding(content)

        lstm_output, lstm_hidden = self.lstm(self.dropout_1(embed_vector))

        last_hidden = torch.squeeze(lstm_hidden[0], 0)

        output = self.linear_out(self.dropout_2(last_hidden))

        

        return output
text_processor = create_text_processor()

ps = nltk.PorterStemmer()

wn = nltk.WordNetLemmatizer()



train_X, test_X, train_y = load_data()



train_X_tensor = torch.tensor(train_X, dtype=torch.long)

train_y_tensor = torch.tensor(train_y[:, np.newaxis], dtype=torch.float32)

test_X_tensor = torch.tensor(test_X, dtype=torch.long)
test_loader = DataLoader(test_X_tensor, batch_size=args['batch_size'], shuffle=False, num_workers=8)



splits = list(StratifiedKFold(n_splits=args['n_splits'], shuffle=True, random_state=args['seed']).split(train_X_tensor, train_y))



val_preds = np.zeros(train_y.shape)

test_preds = []

for idx, (train_idx, val_idx) in enumerate(splits):

    print('Train Fold {}/{}'.format(idx, args['n_splits']))

    

    train_dataset = TensorDataset(train_X_tensor[train_idx], train_y_tensor[train_idx])

    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=8)



    valid_dataset = TensorDataset(train_X_tensor[val_idx], train_y_tensor[val_idx])

    valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=8)

    

    model = Net(args['hidden_dim'], args['target_dim'])

    model.cuda()

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args['learning_rate'], weight_decay=0.0)

    

    model.zero_grad()

    for epoch in range(args['n_epoch']):

        print('Epoch {}/{}'.format(epoch, args['n_epoch']))

        model.train()

        tr_loss = 0.0

        

        for index, (features, label) in enumerate(train_loader):

            features = features.cuda()

            label = label.cuda()

            preds = model(features)

            loss = criterion(preds, label)

            

            loss.backward()

            optimizer.step()

            model.zero_grad()



            tr_loss += loss.item()



        train_loss = tr_loss / len(train_loader)

        print(f"---Train loss {train_loss}")

        

        # Evaluation

        model.eval()

        vl_loss = 0.0

        fold_val_preds = None

        for features, label in valid_loader:

            features = features.cuda()

            label = label.cuda()

            

            preds = model(features)

            loss = criterion(preds, label)

            vl_loss += loss.item()

            

            if epoch == args['n_epoch'] - 1:

                if fold_val_preds is None:

                    fold_val_preds = torch.sigmoid(preds).detach().cpu().numpy().squeeze()

                else:

                    fold_val_preds = np.append(fold_val_preds, torch.sigmoid(preds).detach().cpu().numpy().squeeze(), axis=0)

            

        valid_loss = vl_loss / len(valid_loader)

        print(f"---Valid loss {valid_loss}")

    

    val_preds[val_idx] = fold_val_preds

        

    # Test

    model.eval()

    fold_test_preds = None

    for features in test_loader:

        features = features.cuda()

        preds = model(features)

        

        if fold_test_preds is None:

            fold_test_preds = torch.sigmoid(preds).detach().cpu().numpy().squeeze()

        else:

            fold_test_preds = np.append(fold_test_preds, torch.sigmoid(preds).detach().cpu().numpy().squeeze(), axis=0)

            

    test_preds.append(fold_test_preds)
# Search the best threshold

threshold, val_score = search_threshold(train_y, val_preds)

print('Val Score: {} - Best threshold: {}'.format(val_score, threshold))

# Get mean predicted value of all folds, then convert to label

test_preds_mean = np.mean(test_preds, axis=0)

pred_result = (test_preds_mean > threshold).astype(int)



# Export submission file

submit_df = pd.read_csv(os.path.join(args['data_dir'], 'sample_submission.csv'))

submit_df["target"] = pred_result

submit_df.to_csv(os.path.join(args['output_dir'], 'submission.csv'), index=False)