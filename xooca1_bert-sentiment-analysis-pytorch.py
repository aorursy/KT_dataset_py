# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install pytorch-transformers
import pandas as pd

import numpy as np



#from __future__ import print_function, division

import torch.nn as nn

import torch

from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM,BertConfig

import logging



#from __future__ import print_function, division



import torch.optim as optim

from torch.optim import lr_scheduler

import time

import os

import copy

import gc

from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
!ls ../input
!ls ../input/pretrained-bert-including-scripts/uncased_l-24_h-1024_a-16/uncased_L-24_H-1024_A-16
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

sample = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')

#sample = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')

#sample = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')
pd.options.display.max_columns = 100
train = train[0:50000] 

test = test[0:5000] 
BERT_MODEL_PATH = '../input/bert-uncased-large-pytorch/bert-large-uncased-pytorch_model.bin'

BERT_CONFIG = '../input/bert-uncased-large-pytorch/bert-large-uncased-config.json'

#BERT_TOKEN = '../input/bert-uncased-large-pytorch/bert-large-uncased-vocab.txt'

BERT_TOKEN = '../input/pretrained-bert-including-scripts/uncased_l-24_h-1024_a-16/uncased_L-24_H-1024_A-16/vocab.txt'

BERT_ALL_MODEL = '../input/pretrained-bert-including-scripts/uncased_l-24_h-1024_a-16/uncased_L-24_H-1024_A-16'

BERT_CONFIG_NEW = '../input/pretrained-bert-including-scripts/uncased_l-24_h-1024_a-16/uncased_L-24_H-1024_A-16/bert_config.json'
class BertLayerNorm(nn.Module):

        def __init__(self, hidden_size, eps=1e-12):

            """Construct a layernorm module in the TF style (epsilon inside the square root).

            """

            super(BertLayerNorm, self).__init__()

            self.weight = nn.Parameter(torch.ones(hidden_size))

            self.bias = nn.Parameter(torch.zeros(hidden_size))

            self.variance_epsilon = eps



        def forward(self, x):

            u = x.mean(-1, keepdim=True)

            s = (x - u).pow(2).mean(-1, keepdim=True)

            x = (x - u) / torch.sqrt(s + self.variance_epsilon)

            return self.weight * x + self.bias

        



class BertForSequenceClassification(nn.Module):

    def __init__(self, modelpath,num_labels=2):

        super(BertForSequenceClassification, self).__init__()

        self.num_labels = num_labels

        self.bert = BertModel.from_pretrained(modelconfig.modelpath)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, num_labels)

        nn.init.xavier_normal_(self.classifier.weight)

        

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        return logits

    def freeze_bert_encoder(self):

        for param in self.bert.parameters():

            param.requires_grad = False

    

    def unfreeze_bert_encoder(self):

        for param in self.bert.parameters():

            param.requires_grad = True
class Review_Dataset(Dataset):

    def __init__(self,config,review_label_list,tokenizer):   

        self.review_label_list = review_label_list

        #self.max_seq_length = max_seq_length

        self.modelconfig = config

        self.tokenizer = tokenizer

        

    def __getitem__(self,index):

        #print(self.modelconfig.tokenizer)

        tokenized_review = self.tokenizer.tokenize(self.review_label_list[0][index])

        if len(tokenized_review) > self.modelconfig.max_seq_length:

            tokenized_review = tokenized_review[:self.modelconfig.max_seq_length]

        ids_review  = tokenizer.convert_tokens_to_ids(tokenized_review)

        padding = [0] * (self.modelconfig.max_seq_length - len(ids_review))

        ids_review += padding

        assert len(ids_review) == self.modelconfig.max_seq_length

        ids_review = torch.tensor(ids_review)

        labels = self.review_label_list[1][index]        

        list_of_labels = [torch.from_numpy(np.array(labels))]

        return ids_review, list_of_labels[0]

    

    def __len__(self):

        return len(self.review_label_list[0])
def train_model(model, criterion, optimizer, scheduler,num_epochs=25):

    since = time.time()

    

    print('starting')

    best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = 100

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)

        for phase in ['train', 'val']:

            if phase == 'train':

                scheduler.step()

                model.train()

            else:

                model.eval()

            running_loss = 0.0

            labels_corrects = 0

            for inputs, labels in dataloaders_dict[phase]:

                inputs = inputs.to(device) 

                labels = labels.to(device)

                labels = labels.float()

                #inputs = inputs.float()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    #print(model)

                    outputs = model(inputs)

                    outputs = F.softmax(outputs,dim=1)

                    outputs = outputs.float()

                    #loss = criterion(outputs, torch.max(labels.float(), 1)[1])

                    #print(labels)

                    #print(outputs)

                    loss = criterion(outputs, labels)

                    if phase == 'train':                       

                        loss.backward()

                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                labels_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(labels, 1)[1])

            epoch_loss = running_loss / dataset_sizes[phase]

            labels_acc = labels_corrects.double() / dataset_sizes[phase]

            print('{} total loss: {:.4f} '.format(phase,epoch_loss ))

            print('{} label_acc: {:.4f}'.format(phase, labels_acc))

            if phase == 'val' and epoch_loss < best_loss:

                print('saving with loss of {}'.format(epoch_loss),'improved over previous {}'.format(best_loss))

                best_loss = epoch_loss

                best_model_wts = copy.deepcopy(model.state_dict())

                torch.save(model.state_dict(), 'bert_model_test.pth')

        print()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(float(best_loss)))

    model.load_state_dict(best_model_wts)

    return model
from sklearn.model_selection import train_test_split

def divide_dataset(maindf,reviewcol,labelcollist,batch_size,tokenizer,testsize=0.20):

    review= maindf[reviewcol]

    label = maindf[labelcollist]

    review_train, review_test, label_train, label_test = train_test_split(review, label, test_size=testsize, random_state=42) 

    #print(review_train.shape, review_test.shape, label_train.shape, label_test.shape)

    review_train = review_train.values.tolist()

    review_test = review_test.values.tolist()

    label_train = label_train.values.tolist()

    label_test = label_test.values.tolist()

    #print(review_train[0:4])

    #print(review_test[0:4])

    #print(label_train[0:4])

    #print(label_test[0:4])

    

    #label_train = pd.get_dummies(y_train).values.tolist()

    #label_test = pd.get_dummies(y_test).values.tolist()

    #train_lists = [review_train, label_train]

    #test_lists = [label_train, label_test]

    training_dataset = Review_Dataset(modelconfig,[review_train, label_train],tokenizer )

    test_dataset = Review_Dataset(modelconfig,[review_test, label_test],tokenizer)

    

    dataloaders_dict = {'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0),

                   'val':torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

                   }

    dataset_sizes = {'train':len(review_train),'val':len(review_test)}

    gc.enable()

    del maindf,training_dataset,test_dataset,review_train, review_test, label_train, label_test

    gc.collect()

    return dataloaders_dict,dataset_sizes
class Config(dict):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        for k, v in kwargs.items():

            setattr(self, k, v)

    

    def set(self, key, val):

        self[key] = val

        setattr(self, key, val)
modelconfig = Config(

    review_column='comment_text',

    label_columns = ['severe_toxicity','obscene','threat','insult','identity_attack','sexual_explicit'],

    batch_size=32,

    test_size = 0.2,

    max_seq_length = 256,

    num_epochs = 3,

    modelpath = 'bert-base-uncased',

    vocabpath = 'bert-base-uncased',

    

    

)
tokenizer = BertTokenizer.from_pretrained(modelconfig.vocabpath)

dataloaders_dict,dataset_sizes = divide_dataset(train,modelconfig.review_column,modelconfig.label_columns,

                                                modelconfig.batch_size,tokenizer,modelconfig.test_size)
lrlast = .001

lrmain = .00001

config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,

        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)



num_labels = 6

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification(modelpath=modelconfig.modelpath,num_labels=num_labels)

model = model.to(device)

optim1 = optim.Adam(

    [

        {"params":model.bert.parameters(),"lr": lrmain},

        {"params":model.classifier.parameters(), "lr": lrlast},

       

   ])



#optim1 = optim.Adam(model.parameters(), lr=0.001)#,momentum=.9)

# Observe that all parameters are being optimized

optimizer_ft = optim1

criterion = nn.BCEWithLogitsLoss()



exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)
model1 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,

                       num_epochs=10)
train.head()