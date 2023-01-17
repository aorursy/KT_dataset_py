import math

import os

import pickle

from dataclasses import dataclass



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import plotly.express as px

import torch

import torch.nn.functional as F

from sklearn import metrics

from sklearn.model_selection import train_test_split

from torch.nn.utils.rnn import pad_sequence

from torch.utils.data.dataset import Dataset

from transformers import AdamW, BertForSequenceClassification, BertTokenizer

from tqdm import tqdm

from tqdm.notebook import tqdm as tqdm_notebook
data = pd.read_csv('/kaggle/input/imdb-ptbr/imdb-reviews-pt-br.csv', index_col=0)

data.head(1)
data.info()
data.groupby('sentiment').count()
data['text_pt'].apply(len).hist()
test_dev_size = int(0.05*data.shape[0])

train_dev, test = train_test_split(data, test_size=test_dev_size, random_state=42, stratify=data['sentiment'])

train, dev = train_test_split(train_dev, test_size=test_dev_size, random_state=42, stratify=train_dev['sentiment'])

print('Training samples:', train.shape[0])

print('Dev samples:     ', dev.shape[0])

print('Test samples:    ', test.shape[0])


class ImdbPt(Dataset):

    ''' Loads IMDB-pt dataset. 

    

    It will tokenize our inputs and cut-off those that exceed 512 tokens (the pretrained BERT limit)

    '''

    

    def __init__(self, tokenizer, data, cachefile, rebuild=False):

        if os.path.isfile(cachefile) and rebuild is False:

            self.deserialize(cachefile)

        else:

            self.build(tokenizer, data)

            self.serialize(cachefile)

        

    

    def build(self, tokenizer, data):    

        data = data.copy()

    

        tqdm.pandas()

        data['tokenized'] = data['text_pt'].progress_apply(tokenizer.tokenize)

        

        data['input_ids'] = data['tokenized'].apply(

            lambda tokens: tokenizer.build_inputs_with_special_tokens(

                tokenizer.convert_tokens_to_ids(tokens)))

        

        data = data[data['input_ids'].apply(len)<512]

        

        data['labels'] = (data['sentiment'] == 'pos').astype(int)

        

        self.examples = data[['input_ids', 'labels']].to_dict('records')

    

    def __getitem__(self, i):

        if isinstance(i, int):

            return {key: torch.tensor(value) for key, value in self.examples[i].items()}

        else:

            return [{key: torch.tensor(value) for key, value in sample.items()} for sample in self.examples[i]]

     

    def __len__(self):

        return len(self.examples)

    

    def serialize(self, cachefile):

        with open(cachefile, 'wb') as file:

            pickle.dump(self.examples, file)

    

    def deserialize(self, cachefile):

        with open(cachefile, 'rb') as file:

            self.examples = pickle.load(file)

def data_collator(examples, tokenizer):

    data = {}

    data['input_ids'] = pad_sequence(

            [ex['input_ids'] for ex in examples],

            batch_first=True,

            padding_value=tokenizer.pad_token_id)

    data['labels'] = torch.tensor([ex['labels'] for ex in examples])

    

    attention_mask = torch.zeros(data['input_ids'].shape, dtype=torch.long)

    attention_mask[data['input_ids'] != tokenizer.pad_token_id] = 1   

    data['attention_mask'] = attention_mask

    return data

@dataclass

class DataLoader:

    dataset: ImdbPt

    tokenizer: BertTokenizer

    batch_size: int



    def __iter__(self):

        dataset = self.dataset

        tokenizer = self.tokenizer

        batch_size = self.batch_size

        for start in range(0, len(dataset) - batch_size, batch_size):

            yield data_collator(dataset[start: start+batch_size], tokenizer)



    def __len__(self):

        return math.ceil(len(self.dataset)/self.batch_size)
def send_inputs_to_device(inputs, device):

    return {key:tensor.to(device) for key, tensor in inputs.items()}

tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

train_dataset = ImdbPt(tokenizer, train, '/kaggle/working/train.pkl')

dev_dataset   = ImdbPt(tokenizer, dev,   '/kaggle/working/dev.pkl')

test_dataset  = ImdbPt(tokenizer, test,  '/kaggle/working/test.pkl')

print('Preserved: \n\t Train: {:.2f}% \t Dev: {:.2f}% \t Test: {:.2f}%'.format(

    100 * len(train_dataset) / len(train), 

    100 * len(dev_dataset) / len(dev), 

    100 * len(test_dataset) / len(test)))
train_loader = DataLoader(train_dataset, tokenizer, 8)

dev_loader = DataLoader(dev_dataset, tokenizer, 16)

test_loader = DataLoader(test_dataset, tokenizer, 16)

model = BertForSequenceClassification.from_pretrained(

    'neuralmind/bert-base-portuguese-cased')

model.train()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model.to(device)



optimizer = AdamW(model.parameters(), lr=5e-6)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9997)





for param in model.base_model.parameters():

    param.requires_grad = False
   

def evaluate(model, dev_loader, device):

    with torch.no_grad():

        model.eval()

        dev_losses = []

        tp, tn, fp, fn = [], [], [], []

        for inputs in dev_loader:

            inputs = send_inputs_to_device(inputs, device)

            loss, scores = model(**inputs)[:2]

            dev_losses.append(loss.cpu().item())



            _, classification = torch.max(scores, 1)

            labels = inputs['labels']

            tp.append(((classification==1) & (labels==1)).sum().cpu().item())

            tn.append(((classification==0) & (labels==0)).sum().cpu().item())

            fp.append(((classification==1) & (labels==0)).sum().cpu().item())

            fn.append(((classification==0) & (labels==1)).sum().cpu().item())



        tp_s, tn_s, fp_s, fn_s = sum(tp), sum(tn), sum(fp), sum(fn)

        print('Dev loss: {:.2f}; Acc: {:.2f}; tp: {}; tn: {}; fp: {}; fn: {}'.format( 

              np.mean(dev_losses), (tp_s+tn_s)/(tp_s+tn_s+fp_s+fn_s), tp_s, tn_s, fp_s, fn_s))



        model.train()
epoch_bar = tqdm_notebook(range(1))

loss_acc = 0

alpha = 0.95

for epoch in epoch_bar:

    batch_bar = tqdm_notebook(enumerate(train_loader), desc=f'Epoch {epoch}', total=len(train_loader))

    for idx, inputs in batch_bar:

        if (epoch * len(train_loader) + idx) == 800:

            for param in model.base_model.parameters():

                param.requires_grad = True



        inputs = send_inputs_to_device(inputs, device)

        optimizer.zero_grad()

        loss, logits = model(**inputs)[:2]

        

        loss.backward()

        optimizer.step()

        if epoch == 0 and idx == 0:

            loss_acc = loss.cpu().item()

        else:

            loss_acc = loss_acc * alpha + (1-alpha) * loss.cpu().item()

        batch_bar.set_postfix(loss=loss_acc)

        if idx%200 == 0:

            del inputs

            del loss

            evaluate(model, dev_loader, device)



        scheduler.step()

    os.makedirs('/kaggle/working/checkpoints/epoch'+str(epoch))

    model.save_pretrained('/kaggle/working/checkpoints/epoch'+str(epoch))   



with torch.no_grad():

    model.eval()

    pred = []

    labels = []

    for inputs in tqdm_notebook(dev_loader):

        inputs = send_inputs_to_device(inputs, device)

        _, scores = model(**inputs)[:2]

        pred.append(F.softmax(scores, dim=1)[:, 1].cpu())

        labels.append(inputs['labels'].cpu())

pred = torch.cat(pred).numpy()

labels = torch.cat(labels).numpy()

fpr, tpr, thresholds = metrics.roc_curve(labels, pred, pos_label=1)

roc_auc = metrics.auc(fpr, tpr)



fig = px.scatter(

    x=fpr, y=tpr, color=thresholds, 

    labels={'x': 'False positive rate', 'y': 'True positive rate'},

    title='Curva ROC')

fig.show()
acc = []

for th in thresholds:

    acc.append((~((pred > th) ^ (labels == 1))).sum()/len(labels))



fig2 = px.scatter(

    x=thresholds, y=acc, labels={'x': 'threshold', 'y': 'acurácia'}, 

    title='Acurácia em diferentes thresholds')

fig2.show()  

    
with torch.no_grad():

    model.eval()

    pred = []

    labels = []

    for inputs in tqdm_notebook(test_loader):

        inputs = send_inputs_to_device(inputs, device)

        _, scores = model(**inputs)[:2]

        pred.append(F.softmax(scores, dim=1)[:, 1].cpu())

        labels.append(inputs['labels'].cpu())

pred = torch.cat(pred).numpy()

labels = torch.cat(labels).numpy()



print('Acc:', (~((pred > 0.67) ^ (labels == 1))).sum()/len(labels))