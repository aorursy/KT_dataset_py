import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score
import torchtext
from tqdm import tqdm, tqdm_notebook
from nltk import word_tokenize
import random
from torch import optim
text = torchtext.data.Field(lower=True, batch_first=True, tokenize=word_tokenize, fix_length=70)
qid = torchtext.data.Field()
target = torchtext.data.Field(sequential=False, use_vocab=False, is_target=True)
train = torchtext.data.TabularDataset(path='../input/train.csv', format='csv',
                                      fields={'question_text': ('text',text),
                                              'target': ('target',target)})
test = torchtext.data.TabularDataset(path='../input/test.csv', format='csv',
                                     fields={'qid': ('qid', qid),
                                             'question_text': ('text', text)})
text.build_vocab(train, test, min_freq=3)
qid.build_vocab(test)
glove = torchtext.vocab.Vectors('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
tqdm_notebook().pandas() 
text.vocab.set_vectors(glove.stoi, glove.vectors, dim=300)
class TextCNN(nn.Module):
    
    def __init__(self, lm, padding_idx, static=True, kernel_num=128, fixed_length=50, kernel_size=[2, 5, 10], dropout=0.2):
        super(TextCNN, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(lm)
        if static:
            self.embedding.weight.requires_grad = False
        self.embedding.padding_idx = padding_idx
        self.conv = nn.ModuleList([nn.Conv2d(1, kernel_num, (i, self.embedding.embedding_dim)) for i in kernel_size])
        self.maxpools = [nn.MaxPool2d((fixed_length+1-i,1)) for i in kernel_size]
        self.fc = nn.Linear(len(kernel_size)*kernel_num, 1)
        
    def forward(self, input):
        x = self.embedding(input).unsqueeze(1)  # B X Ci X H X W
        x = [self.maxpools[i](torch.tanh(cov(x))).squeeze(3).squeeze(2) for i, cov in enumerate(self.conv)]  # B X Kn
        x = torch.cat(x, dim=1)  # B X Kn * len(Kz)
        y = self.fc(self.dropout(x))
        return y
def search_best_f1(true, pred):
    tmp = [0,0,0] # idx, cur, max
    delta = 0
    for tmp[0] in np.arange(0.1, 0.501, 0.01):
        tmp[1] = f1_score(true, np.array(pred)>tmp[0])
        if tmp[1] > tmp[2]:
            delta = tmp[0]
            tmp[2] = tmp[1]
    return tmp[2], delta

def training(epoch, model, loss_func, optimizer, train_iter):
    e = 0
    
    while e < epoch:
        train_iter.init_epoch()
        losses, preds, true = [], [], []
        for train_batch in tqdm(list(iter(train_iter)), 'epcoh {} training'.format(e)):
            model.train()
            x = train_batch.text.cuda()
            y = train_batch.target.type(torch.Tensor).cuda()
            true.append(train_batch.target.numpy())
            model.zero_grad()
            pred = model.forward(x).view(-1)
            loss = loss_function(pred, y)
            preds.append(torch.sigmoid(pred).cpu().data.numpy())
            losses.append(loss.cpu().data.numpy())
            loss.backward()
#             clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
        train_f1, alpha_train = search_best_f1([j for i in true for j in i], [j for i in preds for j in i])
        print('epcoh {:02} - train_loss {:.4f} - train f1 {:.4f} - delta {:.4f}'.format(
                            e, np.mean(losses), train_f1, alpha_train))
                
        e += 1
    return alpha_train
                
random.seed(1234)
batch_size = 512
train_iter = torchtext.data.BucketIterator(dataset=train,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               sort=False)
def init_network(model, method='xavier', exclude='embedding', seed=123):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    for name, w in model.named_parameters():
        if not exclude in name:
            if 'weight' in name:
                if method is 'xavier':
                    nn.init.xavier_normal_(w)
                elif method is 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0.0)
            else: 
                pass

def print_model(model, ignore='embedding'):
    total = 0
    for name, w in model.named_parameters():
        if not ignore or ignore not in name:
            total += w.nelement()
            print('{} : {}  {} parameters'.format(name, w.shape, w.nelement()))
    print('-------'*4)
    print('Total {} parameters'.format(total))
text.fix_length = 70
model = TextCNN(text.vocab.vectors, padding_idx=text.vocab.stoi[text.pad_token], kernel_size=[1, 2, 3, 5], kernel_num=128, static=False, fixed_length=text.fix_length, dropout=0.1).cuda()
init_network(model)
optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
loss_function = nn.BCEWithLogitsLoss()
print_model(model, ignore=None)
alpha = training(3, model, loss_function, optimizer, train_iter)
def predict(model, test_list):
    pred = []
    with torch.no_grad():
        for test_batch in test_list:
            model.eval()
            x = test_batch.text.cuda()
            pred += torch.sigmoid(model.forward(x).view(-1)).cpu().data.numpy().tolist()
    return pred
test_list = list(torchtext.data.BucketIterator(dataset=test,
                                    batch_size=batch_size,
                                    sort=False,
                                    train=False))
preds = predict(model, test_list)
sub = pd.DataFrame()
sub['qid'] = [qid.vocab.itos[j] for i in test_list for j in i.qid.view(-1).numpy()]
sub['prediction'] = (preds > alpha).astype(int)
sub.head()
sub.to_csv("submission.csv", index=False)