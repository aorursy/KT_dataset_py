import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

import math
import datetime
import string, re
#show all columns
#pd.set_option('display.max_columns', None)
#show all rows
#pd.set_option('display.max_rows', None)
#show textarea length 
#pd.set_option('max_colwidth',100)
#some config values
embedding_size = 300
hidden_size = 256
words_size = 40000
lr = 0.0001
batch_size = 128   
epoch = 5
dropout = 0.1
#define some help function
def cuda_available(tensor):
    if torch.cuda.is_available:
        return tensor.cuda()
    return tensor

def changeTime(allTime):  
    day = 24*60*60  
    hour = 60*60  
    min = 60  
    if allTime <60:          
        return  "%d sec"%math.ceil(allTime)  
    elif  allTime > day:  
        days = divmod(allTime,day)   
        return "%d days, %s"%(int(days[0]),changeTime(days[1]))  
    elif allTime > hour:  
        hours = divmod(allTime,hour)  
        return '%d hours, %s'%(int(hours[0]),changeTime(hours[1]))  
    else:  
        mins = divmod(allTime,min)  
        return "%d mins, %d sec"%(int(mins[0]),math.ceil(mins[1]))
    
def clean(text): 
    ## Remove puncuation
    text = text.translate(string.punctuation)
    ## Convert words to lower case and split them
    text = text.lower()
    ## Remove stop words
    #text = text.split()
    #stops = set(stopwords.words("english"))
    #text = [w for w in text if not w in stops and len(w) >= 3]
    #text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    #text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    #text = re.sub('[^a-zA-Z]',' ', text)
    text = re.sub('  +',' ',text)
    #text = text.split()
    #stemmer = SnowballStemmer('english')
    #stemmed_words = [stemmer.stem(word) for word in text]
    #text = " ".join(stemmed_words)
    return text
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
submission = pd.read_csv('../input/sample_submission.csv')
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)
print("submission shape :", submission.shape)
train_df['clean_text'] = train_df['question_text'].apply(clean)
test_df['clean_text'] = test_df['question_text'].apply(clean)
#split to train and val
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)

#fill up the missing values
train_X = train_df['clean_text'].fillna('_na_').values  #1175509
val_X = val_df['clean_text'].fillna('_na_').values  #130613
test_X = test_df['clean_text'].fillna('_na_').values  #563
#Tokenize the sequences
tokenizer = Tokenizer(num_words=words_size)
tokenizer.fit_on_texts(list(train_X) + list(test_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X =tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

#Get the target values
train_y = train_df['target'].values  #1175509
val_y = val_df['target'].values  #130613
#convert to tensor
train_tensor_X = [torch.tensor(x) for x in train_X]
val_tensor_X = [torch.tensor(x) for x in val_X]
test_tensor_X = [torch.tensor(x) for x in test_X]
train_tensor_y = torch.from_numpy(train_y)
val_tensor_y = torch.from_numpy(val_y)
n_batchs = int(len(train_tensor_X) / batch_size) + 1
print(n_batchs)
#load glove
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))  

word_index = tokenizer.word_index
nb_words = len(word_index)
embedding_matrix = np.zeros((nb_words, embedding_size))
for word, i in word_index.items():
    #1 <= i <= 187159
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i-1] = embedding_vector
#use glove
class GRU_Classifier(nn.Module):  
    def __init__(self, input_size, pretrained_embeddings, embedding_size=300, hidden_size=100,dropout=0.5):
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        super(GRU_Classifier, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        #self.embedding.weight.requires_grad = False
        self.gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, 
                          bidirectional=True, batch_first=True)
        #out 
        self.linear1 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, question, train=True):
        batch = question.size(0)
        question_embed = self.embedding(question)  #bacth x max_len x embedding_size
        #gru_output: batch x max_len x (2 x hidden_size), hidden: num_directions(2) x batch x hidden_size
        hidden = self.init_hidden(batch)
        gru_output, hidden = self.gru(question_embed, hidden) 
        hidden = hidden.transpose(0, 1).contiguous().view(batch, -1)  #batch x (2 x hidden_size)   
        if train:
            hidden = self.dropout(hidden)
        hidden = torch.relu(self.linear1(hidden))  #batch x hidden_size
        if train:
            hidden = self.dropout(hidden)
        return torch.sigmoid(self.linear2(hidden))  
    
    def init_hidden(self, batch_size):
        return cuda_available(torch.zeros(2, batch_size, self.hidden_size))
#use glove
model_gru = GRU_Classifier(nb_words, embedding_matrix, embedding_size=embedding_size, 
                           hidden_size=hidden_size, dropout=dropout).cuda()
criterion = nn.BCELoss() 
optimizer = torch.optim.Adam(model_gru.parameters(), lr=lr)  #non-static
#optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_gru.parameters()), lr=lr)  #static
#train process
train_loss_array = []  #keep total loss
starttime = datetime.datetime.now()
print("Training for %d epochs..." % epoch)
for i in range(epoch):
    train_loss = 0
    for i in range(n_batchs):
        optimizer.zero_grad()
        X = train_tensor_X[i * batch_size:(i + 1) * batch_size]
        y = cuda_available(train_tensor_y[i * batch_size:(i + 1) * batch_size])
        X_lengths = torch.LongTensor([x for x in map(len, X)]).cuda()
        X_tensor = cuda_available(torch.zeros((len(X), X_lengths.max()))).long()
        for idx, (seq, seqlen) in enumerate(zip(X, X_lengths)):
            X_tensor[idx, :seqlen] = seq
        output = model_gru(X_tensor, train=True).squeeze(1)
        loss = criterion(output, y.float())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_loss_array.append(train_loss)
        endtime = datetime.datetime.now()
    print("epoch is %d, train_loss is %.4f, batch is %d, cost time is about %s" % 
          (i, train_loss, batch_size, changeTime((endtime - starttime).seconds)))
print("train finish!")
n_batchs = int(len(val_tensor_X) / batch_size) + 1
print(n_batchs)
#test gru on val_set
best_F1 = 0
best_threshold = 0
for thresh in np.arange(0.1, 0.701, 0.01):
    thresh = np.round(thresh, 2)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(n_batchs):
        X = val_tensor_X[i * batch_size:(i + 1) * batch_size]
        y = cuda_available(val_tensor_y[i * batch_size:(i + 1) * batch_size])
        X_lengths = torch.LongTensor([x for x in map(len, X)]).cuda()
        X_tensor = cuda_available(torch.zeros((len(X), X_lengths.max()))).long()
        for idx, (seq, seqlen) in enumerate(zip(X, X_lengths)):
            X_tensor[idx, :seqlen] = seq
        output = model_gru(X_tensor, train=False)
        predict1 = (output.squeeze(1) > thresh).long()
        TP += ((predict1 == 1) & (y == 1)).sum()
        TN += ((predict1 == 0) & (y == 0)).sum()
        FN += ((predict1 == 0) & (y == 1)).sum()
        FP += ((predict1 == 1) & (y == 0)).sum()
    p = TP.item() / (TP + FP).item()
    r = TP.item() / (TP + FN).item()
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN).item() / (TP + TN + FP + FN).item()
    if F1 > best_F1:
        best_F1 = F1
        best_threshold = thresh
    print("threshold {0}:F1 score={1}, Acc={2}".format(thresh, F1, acc))
print('----------------------------------------------------------')
print('the best_F1 score={0} at threshold {1}'.format(best_F1, best_threshold))
n_batchs = int(len(test_tensor_X) / batch_size) + 1
print(n_batchs)
#predict
predict = []
for i in range(n_batchs):
    X = test_tensor_X[i * batch_size:(i + 1) * batch_size]
    X_lengths = torch.LongTensor([x for x in map(len, X)]).cuda()
    X_tensor = cuda_available(torch.zeros((len(X), X_lengths.max()))).long()
    for idx, (seq, seqlen) in enumerate(zip(X, X_lengths)):
        X_tensor[idx, :seqlen] = seq
    output = model_gru(X_tensor, train=False).squeeze(1)
    result = (output > best_threshold)
    for j in range(len(X)):
        predict.append(result[j].item())
#just see
predict[40:50]
#submit
submission['prediction'] = predict
submission.to_csv('submission.csv', index=False)