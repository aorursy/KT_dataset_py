from IPython.core.display import display, HTML

from IPython.display import Image

import torch

import torch.nn as nn

import torch.nn.functional as F
use_gpu = True

if use_gpu:

    assert torch.cuda.is_available(), 'You either do not have a GPU or is not accessible to PyTorch'
Image("https://pouannes.github.io/fastai-callbacks/fastai_training_loop_vanilla.png",height=400,width=600)
# def train(train_dl, model, epoch, opt, loss_func):

#   for _ in range(epoch):

#     model.train() # model in training mode

#     for xb,yb in train_dl: # loop through the batches 

#       out = model(xb) # forward pass 

#       loss = loss_func(out, yb) # compute the loss

#       loss.backward() # backward pass gradient 

#       opt.step() # update the gradient 

#       opt.zero_grad() 
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def sigmoid(x):

    return 1 / (1 + np.exp(-x))
from torchtext.data import Field

tokenize = lambda x: x.split()

txt_field = Field(sequential=True, tokenize=tokenize, lower=True)



label_field = Field(sequential=False, use_vocab=False)
train=pd.read_csv('/kaggle/input/usinlppracticum/imdb_train.csv')

train.head()
train['sentiment']=train['sentiment'].map({'negative':0,'positive':1})
print({'negative':0,'positive':1})
train.head()
train.to_csv('train.csv',index=False)
from torchtext.data import TabularDataset

tv_datafields = [

                 ("review", txt_field), ("sentiment", label_field)]



trn = TabularDataset( path="train.csv", # the root directory where the data lies

#                train='imdb_train.csv',

               format='csv',

               skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!

               fields=tv_datafields)
trn,val = trn.split(split_ratio=0.9, stratified=False, strata_field='sentiment', random_state=None)
pd.read_csv("/kaggle/input/usinlppracticum/imdb_test.csv").head()
tst_datafields = [("id", None), # we won't be needing the id, so we pass in None as the field

                 ("review", txt_field)

                 ]

tst = TabularDataset(

        path="/kaggle/input/usinlppracticum/imdb_test.csv", # the file path

        format='csv',

        skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!

        fields=tst_datafields)
len(trn), len(val),len(tst)
trn.fields.items()
ex = trn[1]

type(ex)
trn[0].__dict__.keys()
trn[0].__dict__['review'][:10]
ex.review
ex.sentiment
from torchtext import vocab

# specify the path to the localy saved vectors

vec = vocab.Vectors('/kaggle/input/glove6b/glove.6B.300d.txt',cache= '/kaggle/working')

# build the vocabulary using train and validation dataset and assign the vectors

# print('creating Vocab')

txt_field.build_vocab(trn, max_size=60000, vectors=vec)

# build vocab for labels

label_field.build_vocab(trn)
pre_trained_vectors=txt_field.vocab.vectors

pre_trained_vectors.shape
txt_field.vocab.vectors[txt_field.vocab.stoi['the']].shape
txt_field.vocab.freqs.most_common(14)
type(txt_field.vocab.itos), type(txt_field.vocab.stoi), len(txt_field.vocab.itos), len(txt_field.vocab.stoi.keys()),
txt_field.vocab.stoi['and'], txt_field.vocab.itos[4]
from torchtext.data import Iterator,BucketIterator

traindl, valdl = BucketIterator.splits(datasets=(trn, val), # specify train and validation Tabulardataset

                                            batch_sizes=(64,64),  # batch size of train and validation

                                            sort_key=lambda x: len(x.review), # on what attribute the text should be sorted

                                            device=None, # -1 mean cpu and 0 or None mean gpu

                                            sort_within_batch=False, 

                                            repeat=False)
test_iter = Iterator(tst, batch_size=64, sort=False, sort_within_batch=False, repeat=False)
print(len(traindl), len(valdl))
batch = next(iter(traindl)) # BucketIterator return a batch object

print(type(batch))
print(batch.sentiment) # labels of the batch

# tensor([ 0,  0,  0], device='cuda:0')
print(batch.review.shape) # text index and length of the batch
class BatchGenerator:

    def __init__(self, dl, x_field, y_field):

        self.dl, self.x_field, self.y_field = dl, x_field, y_field

        

    def __len__(self):

        return len(self.dl)

    

    def __iter__(self):

        for batch in self.dl:

            X = getattr(batch, self.x_field)

            if self.y_field is not None:

                y = getattr(batch, self.y_field)

            else:

                y = torch.zeros((1))

            y= torch.tensor(y[:, np.newaxis], dtype=torch.float32)

            if use_gpu:

                yield (X.cuda(), y.cuda())

            else:

                yield (X, y)

          
train_batch_it = BatchGenerator(traindl, 'review', 'sentiment')

val_batch_it = BatchGenerator(valdl, 'review', 'sentiment')

test_dl = BatchGenerator(test_iter, "review", None)

# print(next(iter(train_batch_it))shape)
vocab_size = len(txt_field.vocab)

n_out = 1

vocab_size
trn.fields['review'].vocab.vectors.shape
# class SimpleLSTMBaseline(nn.Module):

#     def __init__(self, hidden_dim, emb_dim=300,pretrained_vec=None,

#                  spatial_dropout=0.05, recurrent_dropout=0.1, num_linear=2):

#         super().__init__() # don't forget to call this!

#         self.embedding = nn.Embedding(len(txt_field.vocab), emb_dim)

#         self.embedding.weight.data.copy_(pretrained_vec) # load pretrained vectors

#         self.embedding.weight.requires_grad = False # make embedding non trainable

#         self.embedding_dropout = nn.Dropout2d(0.1)

#         self.lstm_1 = nn.LSTM(emb_dim, hidden_dim, bidirectional=True)

#         self.lstm_2 = nn.LSTM(hidden_dim*2, hidden_dim, bidirectional=True)

#         self.linear = nn.Linear(hidden_dim, hidden_dim//4)

#         self.relu = nn.ReLU()

#         self.dropout = nn.Dropout(0.1)

#         self.out = nn.Linear(hidden_dim//4, 1)

# #         self.predictor = nn.Linear(hidden_dim, 1)

    

#     def forward(self, seq):

#         h_embedding = self.embedding(x)

#         hdn, _ = self.encoder(self.embedding(seq))

#         feature = hdn[-1, :, :]

#         for layer in self.linear_layers:

#             feature = layer(feature)

#         preds = self.predictor(feature)

#         return preds
class SimpleLSTM(nn.Module):

    def __init__(self):

        super(SimpleLSTM, self).__init__()

        

        hidden_size = 128

        max_features,embed_size=60002,300

        self.embedding = nn.Embedding(max_features, embed_size)

        self.embedding.weight = nn.Parameter(torch.tensor(pre_trained_vectors, dtype=torch.float32))

        self.embedding.weight.requires_grad = False

        

#         self.embedding_dropout = nn.Dropout2d(0.1)

        self.lstm_1 = nn.LSTM(embed_size, hidden_size, bidirectional=True)

        self.lstm_2 = nn.LSTM(hidden_size*2, hidden_size, bidirectional=True)

        self.linear = nn.Linear(hidden_size*6, 16)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.1)

        self.out = nn.Linear(16, 1)

    def forward(self, x):

        h_embedding = self.embedding(x)

#             h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))



        h_lstm_1, _ = self.lstm_1(h_embedding)

        h_lstm_2, _ = self.lstm_2(h_lstm_1)



        avg_pool = torch.mean(h_lstm_1, 0)

        max_pool, _ = torch.max(h_lstm_2, 0)



        conc = torch.cat((h_lstm_2[-1,:,:], avg_pool, max_pool), 1)# 128*2, 128*2, 128*2

        conc = self.relu(self.linear(conc))

        conc = self.dropout(conc)

        out = self.out(conc)



        return out
x,y=next(iter(train_batch_it))

x.shape
m=SimpleLSTM()

m.cuda()

m
em_sz = 300

# nh = 256/

model = SimpleLSTM()

if use_gpu:

    model = model.cuda()

print(model)
def model_size(model: torch.nn)->int:

    """

    Calculates the number of trainable parameters in any model

    

    Returns:

        params (int): the total count of all model weights

    """

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())

#     model_parameters = model.parameters()

    params = sum([np.prod(p.size()) for p in model_parameters])

    return params



print(f'{model_size(model)/10**6} million parameters')
from torch import optim

opt = optim.Adam(model.parameters(), lr=1e-2)

loss_func = nn.BCEWithLogitsLoss().cuda()

epochs = 10
next(iter(traindl))
model=SimpleLSTM()

model.cuda()
x.shape
x,y=next(iter(train_batch_it))

model(x.cuda()).shape
Image("https://pouannes.github.io/fastai-callbacks/fastai_training_loop_vanilla.png",height=400,width=600)


from tqdm import tqdm

for epoch in tqdm(range(1, epochs + 1)):

    running_loss = 0.0

    model.train() # turn on training mode

    for x, y in train_batch_it: # thanks to our wrapper, we can intuitively iterate over our data!

        preds = model(x)

#         print(preds.dtype,y.dtype)

        loss = loss_func(preds, y)

        loss.backward()

        opt.step()

        opt.zero_grad()

        #---

        running_loss += loss.item() * x.size(0)

    epoch_loss = running_loss / len(trn)

    # calculate the validation loss for this epoch

    val_loss = 0.0

    val_acc=0

    model.eval() # turn on evaluation mode

    with torch.no_grad():

        for x, y in val_batch_it:

            preds = model(x)

            y_pred = sigmoid(preds.detach().cpu().numpy()).astype(int)

            loss = loss_func(preds, y)

            val_loss += loss.item() * x.size(0)

#             out = (sigmoid(preds>0.5))

            

            val_acc += (y_pred == y.cpu().numpy().astype(int)).sum().item()

        val_loss /= len(val)

        val_acc=val_acc/len(val)

    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f},val_acc:{:.4f}'.format(epoch, epoch_loss, val_loss,val_acc))
next(iter(test_dl))
model.eval()
with torch.no_grad():

    test_preds = []

    for x, y in tqdm(test_dl):

        preds = model(x)

        # if you're data is on the GPU, you need to move the data back to the cpu

        # preds = preds.data.cpu().numpy()

        preds = preds.data.cpu().numpy()

        # the actual outputs of the model are logits, so we need to pass these values to the sigmoid function

        preds = 1 / (1 + np.exp(-preds))

    #     print(preds.shape)

        test_preds.append(preds[:,0])

    test_preds = np.hstack(test_preds)
test_df = pd.read_csv("/kaggle/input/usinlppracticum/imdb_test.csv")

test_df['sentiment'] = np.where(test_preds>0.5,'positive','negative')

test_df.head()
test_df.sentiment.value_counts()