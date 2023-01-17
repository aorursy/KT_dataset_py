#!pip install torch==1.2.0
import numpy as np

import pandas as pd

import re

from nltk.corpus import stopwords

import time

import datetime



from sklearn.metrics import roc_auc_score



import torch

from torch import nn, optim

from torch.utils.data import TensorDataset, DataLoader, Dataset

from torch.autograd import Variable



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
STOPWORDS = set([])  # set(stopwords.words('english'))

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-z\s]', '', text)

    text = ' '.join([word for word in text.split() if word not in STOPWORDS])

    return text



def tokenize(text, word_to_idx):

    tokens = []

    for word in text.split():

        tokens.append(word_to_idx[word])

    return tokens





def pad_and_truncate(messages, max_length=30):

    features = np.zeros((len(messages), max_length), dtype=int)

    for i, sms in enumerate(messages):

        if len(sms):

            features[i, -len(sms):] = sms[:max_length]

    return features
data = pd.read_csv('../input/gc4classes/gender-classifier-data.csv')

data = pd.concat([data.gender,data.description],axis=1)

data.dropna(inplace=True,axis=0)

data.gender = [1 if each == "female" else 0 for each in data.gender] 


data_gender = []



for t in data.gender:

    v = np.zeros(2)

    if t==0:

        v[0] = 1

    else:

        v[1] = 1

    data_gender.append(v)
data_gender
data
data.description = data.description.apply(clean_text)
words = set((' '.join(data.description)).split())

print(len(words))
word_to_idx = {word: i for i, word in enumerate(words, 1)}

tokens = data.description.apply(lambda x: tokenize(x, word_to_idx))

inputs = pad_and_truncate(tokens)



#labels = np.array((data.gender).astype(int))



labels = np.array(data_gender)

#labels = data.gender
labels.shape
print(inputs[:2])

print(labels[:2])
VOCAB_SIZE = int(inputs.max()) + 1

# Training params

EPOCHS = 1000

CLIP = 5 # gradient clipping - to avoid gradient explosion (frequent in RNNs)

lr = 0.1

BATCH_SIZE = 32



# Model params

EMBEDDING_DIM = 100

HIDDEN_DIM = 10

DROPOUT = 0.2
labels = torch.tensor(labels)

inputs = torch.tensor(inputs)



pct_test = 0.2



train_labels = labels[:-int(len(labels)*pct_test)]

train_inputs = inputs[:-int(len(labels)*pct_test)]



test_labels = labels[-int(len(labels)*pct_test):]

test_inputs = inputs[-int(len(labels)*pct_test):]
test_labels
class TextSentiment(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):

        super().__init__()

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)

        self.fc = nn.Linear(embed_dim, num_class)

        self.init_weights()

        self.sof = nn.Softmax()



    def init_weights(self):

        initrange = 0.5

        self.embedding.weight.data.uniform_(-initrange, initrange)

        self.fc.weight.data.uniform_(-initrange, initrange)

        self.fc.bias.data.zero_()



    def forward(self, text, offsets):

        embedded = self.embedding(text, offsets)

        return self.sof(self.fc(embedded))
model = TextSentiment(vocab_size=VOCAB_SIZE, embed_dim=EMBEDDING_DIM, num_class=2)

model.to('cuda:0')
class dataset(Dataset):

    def __init__(self, inputs, labels):

        self.inputs =inputs

        self.labels = labels



    def __len__(self):

        return len(self.labels)



    def __getitem__(self, idx):

        return self.inputs[idx],self.labels[idx]
train_dataloader = DataLoader(dataset(train_inputs,train_labels),

                               batch_size=2048, 

                               shuffle=True,

                               num_workers=0, 

                               pin_memory=True)



test_dataloader = DataLoader(dataset(test_inputs,test_labels),

                               batch_size=2048, 

                               shuffle=True,

                               num_workers=0, 

                               pin_memory=True)
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.SGD(model.parameters(), lr=lr)
def train(epoch,dloader):

    for step, (x,y) in enumerate(dloader):

        data = Variable(x).cuda()

        target = Variable(y).cuda()

        

        h = torch.Tensor(np.zeros((BATCH_SIZE, HIDDEN_DIM)))

        output = model(data,None)

        loss = criterion(output, target.float())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if step==0:

            start = time.time()

            ti = 0

        elif step==50:

            ti = time.time()-start #total time = ti*(length/100)

            #print(ti)

            ti = ti*(len(dloader)/50)

        if step % 50 == 0:

            second = ti*(((len(dloader)-step)/len(dloader)))#*(5-epoch)*(4-fnum)

            print('Ep: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}\t Remain : {} '.

                     format(epoch+1, 

                            step * len(data), 

                            len(dloader.dataset),

                            50.*step/len(dloader), 

                            loss.data.item(),

                            datetime.timedelta(seconds = int(second))))

        torch.cuda.empty_cache()

    #print("Finish")
def val(dloader):

    los = []

    acc_num = 0

    for step, (x, y) in enumerate(dloader):

        data = Variable(x).cuda()

        target = Variable(y).cuda()

        h = torch.Tensor(np.zeros((BATCH_SIZE, HIDDEN_DIM)))

        with torch.no_grad():

            output = model(data,None)

        

        loss = criterion(output, target.float())

        los.append(loss.item())

        

        out = np.argmax(target.cpu().data.numpy().squeeze(),axis = 1)

        pre = np.argmax(output.cpu().data.numpy().squeeze(),axis = 1)

        #print(pre)

        #print(out)

        acc_num += (out == pre).sum()

        

        if step %50 == 0:

            print('[{}/{} ({:.1f}%)]'.format(step * len(data), 

                                        len(dloader.dataset),

                                        50.*step/len(dloader)))



        torch.cuda.empty_cache()

    print("Acc : [{}/{} ({:.1f}%)]".format(acc_num,

                                         len(dloader.dataset),

                                         100.*acc_num/len(dloader.dataset)))

    los = np.array(los)

    avg_val_loss = los.sum()/len(los)

    print("Avg val loss: avg_val_loss {:.8f}".format(loss))

for epoch in range(EPOCHS):

    if epoch %100==0:

        lr = lr*0.9

        optimizer = optim.SGD(model.parameters(), lr=lr)

    model.train()

    train(epoch, train_dataloader)

    model.eval()

    val(test_dataloader)
dataT = pd.read_csv('../input/gc4classes/gender-classifier-test.csv')

dataT.dropna(inplace=True,axis=0)

dataT.description  
dataT.description = data.description.apply(clean_text)
word_to_idx = {word: i for i, word in enumerate(words, 1)}

tokens = data.description.apply(lambda x: tokenize(x, word_to_idx))

inputs = pad_and_truncate(tokens)
len(inputs)
Test_dataloader = DataLoader(dataset(inputs,[[0]*len(inputs)][0]),

                               batch_size=2048, 

                               shuffle=True,

                               num_workers=0, 

                               pin_memory=True)
def test(dloader):

    los = []

    acc_num = 0

    ans = []

    for step, (x, y) in enumerate(dloader):

        data = Variable(x).cuda()

        target = Variable(y).cuda()

        h = torch.Tensor(np.zeros((BATCH_SIZE, HIDDEN_DIM)))

        with torch.no_grad():

            output = model(data,None)

        

        #loss = criterion(output, target.float())

        #los.append(loss.item())

        

        #out = np.argmax(target.cpu().data.numpy().squeeze(),axis = 1)

        #print(output)

        pre = np.argmax(output.cpu().data.numpy().squeeze(),axis = 1)

        #print(pre)

        #print(out)

        #acc_num += (out == pre).sum()

        for id in range(len(pre)):

            ans.append([pre[id]])

        

        if step %50 == 0:

            print('[{}/{} ({:.1f}%)]'.format(step * len(data), 

                                        len(dloader.dataset),

                                        50.*step/len(dloader)))



        torch.cuda.empty_cache()



    return ans

ans = test(Test_dataloader)
Ans = [[i,ans[i][0]] for i in range(len(ans))]
len(Ans)
Ans[:5]
sub =  pd.DataFrame(Ans)

sub = sub.rename(index=str, columns={0: "no.", 1: "gender"})

words = set((' '.join(data.description)).split())

print(len(words))
sub
sub.to_csv('submission.csv', index=False)