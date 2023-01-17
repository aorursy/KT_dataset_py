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
import datetime
import nltk
import operator 
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
train_df = pd.read_csv("../input/train.csv")
train_df_len = train_df.shape[0]
print('train data length: {}'.format(train_df_len)) # 1306122
train_df.head()
# statistis of target 0 and 1
t0, t1 = len(train_df[train_df.target == 0]), len(train_df[train_df.target == 1])
t0_pct, t1_pct = t0 / train_df_len * 100, t1 / train_df_len * 100
print('target 0 vs 1 = {} vs {}, {:.2f}% vs {:.2f}%'.format(t0, t1, t0_pct, t1_pct))
test_df = pd.read_csv("../input/test.csv")
print('test data length: {}'.format(test_df.shape[0]))
test_df.head()
sample_df = pd.read_csv('../input/sample_submission.csv')
print('sample submission length: {}'.format(sample_df.shape[0]))
sample_df.head()
del sample_df
# Contractions corrections
contraction_dict = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot",
    "'cause": "because", "could've": "could have", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not",
    "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "he'd": "he would", "he'll": "he will", "he's": "he is",
    "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
    "how's": "how is", "I'd": "I would", "I'd've": "I would have",
    "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
    "I've": "I have", "i'd": "i would", "i'd've": "i would have",
    "i'll": "i will",  "i'll've": "i will have", "i'm": "i am",
    "i've": "i have", "isn't": "is not", "it'd": "it would",
    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
    "it's": "it is", "let's": "let us", "ma'am": "madam",
    "mayn't": "may not", "might've": "might have", "mightn't": "might not",
    "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
    "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
    "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
    "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
    "she'll've": "she will have", "she's": "she is", "should've": "should have",
    "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
    "so's": "so as", "this's": "this is", "that'd": "that would",
    "that'd've": "that would have", "that's": "that is", "there'd": "there would",
    "there'd've": "there would have", "there's": "there is", "here's": "here is",
    "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
    "they'll've": "they will have", "they're": "they are", "they've": "they have",
    "to've": "to have", "wasn't": "was not", "we'd": "we would",
    "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
    "we're": "we are", "we've": "we have", "weren't": "were not",
    "what'll": "what will", "what'll've": "what will have", "what're": "what are",
    "what's": "what is", "what've": "what have", "when's": "when is",
    "when've": "when have", "where'd": "where did", "where's": "where is",
    "where've": "where have", "who'll": "who will", "who'll've": "who will have",
    "who's": "who is", "who've": "who have", "why's": "why is",
    "why've": "why have", "will've": "will have", "won't": "will not",
    "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
    "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
    "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
    "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
    "you'll've": "you will have", "you're": "you are", "you've": "you have"
}
def clean_contractions(text, contraction_dict):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([contraction_dict[t] if t in contraction_dict else t for t in text.split(" ")])
    return text
# special characters
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_dict = {
    "‘": "'",    "₹": "e",      "´": "'", "°": "",         "€": "e",
    "™": "tm",   "√": " sqrt ", "×": "x", "²": "2",        "—": "-",
    "–": "-",    "’": "'",      "_": "-", "`": "'",        '“': '"',
    '”': '"',    '“': '"',      "£": "e", '∞': 'infinity', 'θ': 'theta',
    '÷': '/',    'α': 'alpha',  '•': '.', 'à': 'a',        '−': '-',
    'β': 'beta', '∅': '',       '³': '3', 'π': 'pi'
}
def clean_special_chars(text, punct, punct_dict):
    for p in punct_dict:
        text = text.replace(p, punct_dict[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text
stopwords = nltk.corpus.stopwords.words('english')
def preprocess(df, contraction_dict, punct, punct_dict):
    texts = df.question_text
    processed_texts = texts.apply(lambda x: x.lower())
    processed_texts = processed_texts.apply(lambda x: clean_contractions(x, contraction_dict))
    processed_texts = processed_texts.apply(lambda x: clean_special_chars(x, punct, punct_dict))
    processed_texts = processed_texts.apply(lambda x: re.split('\W+', x))
    processed_texts = processed_texts.apply(lambda x: [token for token in x if token not in stopwords])
    df['processed_text'] = processed_texts
#SAMPLE_ROWS_T0 = 575000
#SAMPLE_ROWS_T0 = 1220000 # too many positive data makes test score worse
#SAMPLE_ROWS_T1 = 80000
#SAMPLE_ROWS_T0 = 273200 # too few
#SAMPLE_ROWS_T0 = 547200
#SAMPLE_ROWS_T1 = 80800
SAMPLE_ROWS_T0 = 639190
SAMPLE_ROWS_T1 = 80810
df_t0 = train_df[train_df.target==0].sample(SAMPLE_ROWS_T0)
df_t1 = train_df[train_df.target==1].sample(SAMPLE_ROWS_T1)
preprocess(df_t0, contraction_dict, punct, punct_dict)
df_t0.head()
preprocess(df_t1, contraction_dict, punct, punct_dict)
df_t1.head()
preprocess(test_df, contraction_dict, punct, punct_dict)
test_df.head()
def build_vocab(texts, vocab):
    for word in texts:
        vocab.add(word)
vocab = set()
df_t1.processed_text.apply(lambda x: build_vocab(x, vocab))
df_t0.processed_text.apply(lambda x: build_vocab(x, vocab))
test_df.processed_text.apply(lambda x: build_vocab(x, vocab))
print(len(vocab))
def load_embed(filename, vocab):
    word2vec = {}
    def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')
    f = open(filename, encoding='latin')
    for line in tqdm(f):
        word, coefs = get_coefs(*line.split(" "))
        if word in vocab:
            word2vec[word] = coefs
    f.close()
    return word2vec
#glove = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
paragram =  '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
#wiki_news = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
word2vec = load_embed(paragram, vocab)
len(word2vec), word2vec['add'].shape
# see the train data closely
# min and max number of words in questions
lens_t0 = list(map(len, df_t0.processed_text))
lens_t1 = list(map(len, df_t1.processed_text))
lens_test = list(map(len, test_df.processed_text))
print('min and max words in pos questions: {}, {}'.format(min(lens_t0), max(lens_t0)))
print('min and max words in neg questions: {}, {}'.format(min(lens_t1), max(lens_t1)))
print('min and max words in test questions: {}, {}'.format(min(lens_test), max(lens_test)))
def freq_stats(tag, counts, key, topk, total):
    most_freqs = sorted(counts, key=key, reverse=True)[:topk]
    freqs = [counts[freq] for freq in most_freqs]
    print('{}: best {} frequent word count: {}, '.format(tag, topk, most_freqs),
          'freqs: {}, '.format(freqs),
          'covers: {:.2f}%'.format(sum(freqs)/total*100))
    return max(most_freqs)

from collections import Counter
counts_t0 = Counter(lens_t0)
counts_t1 = Counter(lens_t1)
counts_test = Counter(lens_test)
#topk = 5 # vast majority of questions are covered, but may lose clues to classify correctly
topk = 20
max_t0 = freq_stats('pos', counts_t0, counts_t0.get, topk, SAMPLE_ROWS_T0)
max_t1 = freq_stats('neg', counts_t1, counts_t1.get, topk, SAMPLE_ROWS_T1)
max_test = freq_stats('test', counts_test, counts_test.get, topk, test_df.shape[0])
SEQ_LENGTH = max(max_t0, max_t1, max_test)
SEQ_LENGTH
def build_weights_matrix(word2vec):
    word_to_idx = {}
    weights_matrix = np.zeros((len(word2vec), 300))
    for i, (k, v) in enumerate(word2vec.items()):
        word_to_idx[k] = i
        weights_matrix[i] = v
    return word_to_idx, weights_matrix
word_to_idx, weight_matrix = build_weights_matrix(word2vec)
# the length of word vector: seq_length
def encode_question(word_to_idx, text, seq_length):
    encoded = []
    for word in text[:seq_length]:
        try:
            encoded.append(word_to_idx[word])
        except KeyError:
            # missing words in the table such typos or created words
            continue

    return np.array(encoded, dtype='int_')
# adds padding
def add_padding(numpy_array, seq_length):
    cur_length = numpy_array.shape[0]
    if cur_length < seq_length:
        padding = np.zeros((seq_length-cur_length, ), dtype='int_')
        return np.concatenate((padding, numpy_array))
    else:
        return numpy_array
def create_dataset(texts, label, word_to_idx, seq_length):
    texts_len = len(texts)
    y = np.array([label]*texts_len, dtype='float')
    X = []
    for i, text in enumerate(texts):
        text_array = encode_question(word_to_idx, text, seq_length)
        text_array = add_padding(text_array, seq_length)
        X.append(text_array)
    return np.array(X), y
# splits train data to train and validation
TEST_SIZE = 0.1
train_texts_t0, val_texts_t0 = train_test_split(df_t0.processed_text, test_size=TEST_SIZE)
train_texts_t1, val_texts_t1 = train_test_split(df_t1.processed_text, test_size=TEST_SIZE)
train_X_t0, train_y_t0 = create_dataset(train_texts_t0, 0, word_to_idx, SEQ_LENGTH)
train_X_t1, train_y_t1 = create_dataset(train_texts_t1, 1, word_to_idx, SEQ_LENGTH)
train_X = np.concatenate((train_X_t0, train_X_t1))
train_y = np.concatenate((train_y_t0, train_y_t1))
print('shapes: train_X {}, train_y {}'.format(train_X.shape, train_y.shape))
val_X_t0, val_y_t0 = create_dataset(val_texts_t0, 0, word_to_idx, SEQ_LENGTH)
val_X_t1, val_y_t1 = create_dataset(val_texts_t1, 1, word_to_idx, SEQ_LENGTH)
val_X = np.concatenate((val_X_t0, val_X_t1))
val_y = np.concatenate((val_y_t0, val_y_t1))
print('shapes: val_X {}, val_y {}'.format(val_X.shape, val_y.shape))
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

print(torch.__version__)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)
# creates Tensor datasets
train_set = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
val_set = TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_y))
# creates dataloaders
# hyperparameter for data loading
#  - batch_size: size of one batch
BATCH_SIZE = 200

# make sure to SHUFFLE the training data
train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_set, shuffle=True, batch_size=BATCH_SIZE)
# Only LSTM (2 or 3 layers) model suffered an overfitting problem.
# To avoid the problem, GRU and average pooling layer were added.
# The overfitting got better, but still the problem exists.
class SentimentRNN(nn.Module):
    def __init__(self, weights, n_out, n_hidden, n_layers,
                 bidirectional=False, dropout=0.5, layer_dropout=0.3):
        super(SentimentRNN, self).__init__()

        self.n_out = n_out
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        if bidirectional:
            self.direction = 2
        else:
            self.direction = 1

        num_embeddings, embedding_dim = weights.shape
        
        # embedding layer
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(weights))
        self.embedding.weight.requires_grad = False
        # for some reason from_pretrained doesn't work
        #self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(weights))
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, n_hidden, n_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)
        
        # GRU layer
        self.gru = nn.GRU(embedding_dim, n_hidden, n_layers,
                          batch_first=True, dropout=dropout,
                          bidirectional=bidirectional)
        # Conv1d layer
        self.conv1d = nn.Conv1d(n_hidden*self.direction, (n_hidden*self.direction)//2, 1)
        # Average Pooling layer
        self.avp = nn.AvgPool1d(2)
        # Dropout layer
        self.dropout = nn.Dropout(layer_dropout)
        # Fully-conneted layer
        self.fc = nn.Linear((n_hidden*self.direction)//4*2, n_out)
        
        # Sigmoid activation layer
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        seq_len = x.size(1)
        lstm_hidden, gru_hidden = hidden
        
        embeds = self.embedding(x)
        
        lstm_out, lstm_hidden = self.lstm(embeds, lstm_hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.n_hidden*self.direction, seq_len)
        lstm_out = self.conv1d(lstm_out)
        lstm_out = lstm_out.contiguous().view(-1, seq_len, (self.n_hidden*self.direction)//2)
        lstm_out = self.avp(lstm_out)
        
        gru_out, gru_hidden = self.gru(embeds, gru_hidden)
        gru_out = gru_out.contiguous().view(-1, self.n_hidden*self.direction, seq_len)
        gru_out = self.conv1d(gru_out)
        gru_out = gru_out.contiguous().view(-1, seq_len, (self.n_hidden*self.direction)//2)
        gru_out = self.avp(gru_out)
        
        #out = (lstm_out + gru_out) / 2.0
        out = torch.cat((lstm_out, gru_out), 2)
        out = self.dropout(out)
        out = self.dropout(out)
        out = self.fc(out.float())
        sig_out = self.sig(out)

        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get only last labels
        
        return sig_out, (lstm_hidden, gru_hidden)
    
    def init_hidden(self, batch_size, bidirectional=False):
        weight = next(self.parameters()).data
        # for LSTM (initial_hidden_state, initial_cell_state)
        lstm_hidden = (
            weight.new(self.n_layers*self.direction, batch_size, self.n_hidden).zero_().to(DEVICE),
            weight.new(self.n_layers*self.direction, batch_size, self.n_hidden).zero_().to(DEVICE)
        )
        # for GRU, initial_hidden_state
        gru_hidden = weight.new(self.n_layers*self.direction, batch_size, self.n_hidden).zero_().to(DEVICE)
        return lstm_hidden, gru_hidden
# hyperparameters
n_out = 1
#n_hidden = 512
n_hidden = 256
n_layers = 3
# instantiate the network
net = SentimentRNN(weight_matrix, n_out, n_hidden, n_layers, bidirectional=False).to(DEVICE)
net
# hyperparameters for training
#  - lr: learning rate
#  - epochs: number of epochs
lr = 0.00008
epochs = 10
clip = 5 # gradient clipping
# loss and optimizer functions
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)
# for now, scheduler is not used. (has a bigger step_size than epochs)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
def train(net, criterion, optimizer, train_loader, clip, epoch, epochs, gru=True):
    # paramters for printing
    counter = 0
    print_every = 500

    train_length = len(train_loader)
    
    # initialize hidden state
    hidden = net.init_hidden(BATCH_SIZE)
    
    train_losses = []

    # batch loop
    for inputs, labels in train_loader:
        counter += 1

        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        if gru:
            l_h, g_h = hidden
            # for LSTM
            l_h = tuple([each.data for each in l_h])
            # for GRU
            g_h = g_h.data
            hidden = (l_h, g_h)
        else:
            hidden = tuple([each.data for each in hidden])
        
        # zero accumulated gradients
        net.zero_grad()
        
        # get the output from the model
        outputs, hidden = net(inputs, hidden)

        # calcuate the loss and perform backprop
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient probelm in RNNs/ LSTMs
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            train_losses.append(loss.item())
            print("Epoch: {}/{}...".format(epoch+1, epochs),
                  "Step: {}...".format(counter),
                  "Train Loss: {:.6f}...".format(np.mean(train_losses)),
                  "Time: {}".format(datetime.datetime.now()))
# get validation loss
THRESHOLD = 0.6
def validate(net, criterion, val_loader, epoch, epochs, gru=True):
    hidden = net.init_hidden(BATCH_SIZE)
    val_losses = []
    with torch.no_grad():
        for inputs, labels in val_loader:

            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            if gru:
                val_l_h, val_g_h = hidden
                # for LSTM
                val_l_h = tuple([each.data for each in val_l_h])
                # for GRU
                val_g_h = val_g_h.data
                hidden = (val_l_h, val_g_h)
            else:
                hidden = tuple([each.data for each in hidden])

            outputs, hidden = net(inputs, hidden)
            val_loss = criterion(outputs.squeeze(), labels.float())
            val_losses.append(val_loss.item())

            acc = torch.eq(labels.float(), torch.round(outputs.squeeze())).sum().item()

        print("Epoch: {}/{}...".format(epoch+1, epochs),
              "Val Loss: {:.6f}".format(np.mean(val_losses)),
              "Val Acc: {}/{}".format(acc, BATCH_SIZE),
              "Time: {}".format(datetime.datetime.now()))
def run_train(net,
              criterion, optimizer, scheduler,
              epochs, train_loader, val_loader,
              clip, gru=True):
    for epoch in range(epochs):
        scheduler.step()
        train(net, criterion, optimizer, train_loader, clip, epoch, epochs, gru)
        validate(net, criterion, val_loader, epoch, epochs, gru)
run_train(net, criterion, optimizer, scheduler, epochs, train_loader, val_loader, clip, gru=True)
class QuoraTestDataset(Dataset):
    def __init__(self, df, word_to_idx, seq_length):
        self.word_to_idx = word_to_idx
        self.seq_length = seq_length
        self.data = df
        self.data_len = len(df)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if idx >= self.data_len:
            idx %= self.data_len
        # pre-processed
        tokens = self.data.iloc[idx].processed_text
        # encode to make array of indices
        encoded = encode_question(word_to_idx, tokens, self.seq_length) # numpy array of int
        text_array = add_padding(encoded, self.seq_length)
        return self.data.iloc[idx].qid, torch.from_numpy(text_array)
# create dataset
test_set = QuoraTestDataset(test_df, word_to_idx, SEQ_LENGTH)
TEST_BATCH_SIZE = 30
# create dataloader
test_loader = DataLoader(test_set, shuffle=False, batch_size=TEST_BATCH_SIZE)
def test(net, test_loader, batch_size=TEST_BATCH_SIZE):
    test_l_h, test_g_h = net.init_hidden(batch_size)
    ret_qid = []
    ret_pred = []
    test_len = len(test_loader)
    counter = 0
    with torch.no_grad():
        for qids, inputs in test_loader:
            counter += 1
            inputs = inputs.to(DEVICE)
            
            # for LSTM
            test_l_h = tuple([each.data for each in test_l_h])
            # for GRU
            test_g_h = test_g_h.data

            outputs, (test_l_h, test_g_h) = net(inputs, (test_l_h, test_g_h))
            
            ret_qid.append(qids)
            ret_pred.append(torch.round(outputs.squeeze()).cpu().numpy().astype(int))
            
            if counter % 300 == 0:
                print('{}/{} done'.format(counter, test_len))

    return ret_qid, ret_pred
ret_qid, ret_pred = test(net, test_loader)
ret_qid, ret_pred = np.concatenate(ret_qid), np.concatenate(ret_pred)
submit_df = pd.DataFrame({"qid": ret_qid, "prediction": ret_pred})
submit_df.head()
submit_df[-5:]
submit_df.to_csv("submission.csv", index=False)
