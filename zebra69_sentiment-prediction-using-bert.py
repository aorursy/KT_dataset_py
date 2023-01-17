import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers as ppb

from tqdm import tqdm_notebook as tqdm
import random
import matplotlib.pyplot as plt
import warnings

import os
for dirname, _, filenames in os.walk('/kaggle/input/tweet-sentiment-extraction/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

warnings.simplefilter('ignore')
def label_encoding(sentiment):
    #label encoding
    if sentiment == 'positive':
        return 0
    elif sentiment == 'negative':
        return 1
    else:
        return 2
def tokenize_fn(text):
    #BERT model宣言
    tokenizer = ppb.BertTokenizerFast.from_pretrained('/kaggle/input/bert-base-uncased/')
    if str(text) == 'nan':
        text = ' '
    text = text.lower()
    #tokenize using BERTtokenizer
    text = tokenizer.encode(text,do_lower_case=True)
    
    text = np.pad(text,[0,110-len(text)],'constant')
    return text
class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.tokenized = torch.empty(self.data.shape[0], 110)
        self.labels = torch.empty(self.data.shape[0])
        
        for i in tqdm(range(self.data.shape[0])):
            self.tokenized[i] = torch.from_numpy(tokenize_fn(self.data['text'][i]))
            self.labels[i] = label_encoding(self.data['sentiment'][i])
        self.len = self.tokenized.shape[0]
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        out_data = self.tokenized[idx]
        out_label = self.labels[idx]

        return out_data, out_label
trainset = Mydatasets('/kaggle/input/tweet-sentiment-extraction/train.csv')
testset = Mydatasets('/kaggle/input/tweet-sentiment-extraction/test.csv')
train_size = int(0.8 * len(trainset))
test_size = len(trainset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, test_size])
BATCH_SIZE = 16

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
testloader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)
class BERTmodel(nn.Module):
    def __init__(self,bert):
        super(BERTmodel,self).__init__()
        self.bert = bert
        #in_feature=768 depend on output size of BERT model
        self.cls = nn.Linear(in_features=768,out_features=3)
    
    def forward(self,x,token_type_ids=None,attention_mask=None):
        encoded_layers,_ = self.bert(x,token_type_ids,attention_mask)
        word_vec = encoded_layers[:,0,:].view(-1,768)
        out = self.cls(word_vec)
        return out
def train(net,optimizer,criterion,epochs,batch_size,trainset,valset):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:',device)
    print('---start---')
    net.to(device)
    torch.backends.cudnn.benchmark = True
    
    train_acc_list,train_loss_list = [],[]
    val_acc_list,val_loss_list = [],[]
    
    for epoch in range(epochs):
        for phase in ['train','val']:
            print('---%s_%d---' % (phase,epoch))
            if phase == 'train':
                dataset = trainset
                net.train()
            else:
                dataset = valset
                net.eval()
            epoch_loss = 0.0
            epoch_corrects = 0
            total = 0
            
            for inputs,labels in tqdm(dataset):
                inputs = inputs.long().to(device)
                labels = labels.long().to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    outputs = net(inputs)
                    loss = criterion(outputs,labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                    pred = torch.argmax(outputs,dim=1)
                    total += labels.size(0)
                epoch_corrects += (pred == labels).sum().item()
                epoch_loss += loss.item()
                
            if phase == 'train':
                train_loss_list.append(epoch_loss/len(dataset))
                train_acc_list.append(100 * epoch_corrects / total)
            else:
                val_loss_list.append(epoch_loss/len(dataset))
                val_acc_list.append(100 * epoch_corrects / total)
            print('Loss: %f' % (epoch_loss/len(dataset)))
            print('Accuracy: %f' % (epoch_corrects/total))
            
    return (train_acc_list,train_loss_list),(val_acc_list,val_loss_list)
EPOCHS=20

bert = ppb.BertModel.from_pretrained('/kaggle/input/bert-base-uncased/')
net = BERTmodel(bert)

# No gradient calculation for the 1st to 11th transformer. 
# It should be all calculated, but to reduce the cost of calculation.
for name, param in net.named_parameters():
    param.requires_grad = False
    
# 12th transformer with gradient calculation.
for name, param in net.bert.encoder.layer[-1].named_parameters():
    param.requires_grad = True

#dense layer with gradient calculation.(fine-tuning)
for name, param in net.cls.named_parameters():
    param.requires_grad = True

#param update
optimizer = torch.optim.Adam([{'params': net.bert.encoder.layer[-1].parameters(),'lr':5e-5},
                        {'params': net.cls.parameters(),'lr': 5e-5}], betas=(0.9,0.999))

#loss function
criterion = nn.CrossEntropyLoss() 
train_result,val_result = train(net,optimizer,criterion,EPOCHS,BATCH_SIZE,trainloader,valloader)
def plot_history(index,train_data,val_data,title):
    # Plot the loss in the history
    plt.plot(index, train_data, label='train')
    plt.plot(index, val_data, label='val')
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel(title)
    plt.legend()
index = np.arange(20)
plot_history(index,train_result[0],val_result[0],'acc')
plot_history(index,train_result[1],val_result[1],'loss')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
net.eval()
epoch_corrects = 0
total = 0

for inputs,labels in tqdm(testloader):
    inputs = inputs.long().to(device)
    labels = labels.long().to(device)
    optimizer.zero_grad()
    with torch.set_grad_enabled(False):
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        pred = torch.argmax(outputs,dim=1)
        total += labels.size(0)
    epoch_corrects += (pred == labels).sum().item()

print('Accuracy: %f' % (epoch_corrects/total))