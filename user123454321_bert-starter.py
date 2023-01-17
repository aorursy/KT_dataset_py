import pandas as pd

import re

import string

import os

import torch

import numpy as np

import tqdm
!pip install transformers
from transformers import BertModel, BertTokenizer
path_to_dataset = '/kaggle/input/nlp-getting-started/'
test = pd.read_csv(os.path.join(path_to_dataset, 'test.csv'))

train = pd.read_csv(os.path.join(path_to_dataset, 'train.csv'))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
class Model(torch.nn.Module):

    

    def __init__(self, ):

        

        super(Model, self).__init__()

        self.base_model = BertModel.from_pretrained('bert-base-uncased') # use pre-trained BERT model by HuggingFace

        self.fc1 = torch.nn.Linear(768, 1) # simple logistic regression above the bert model

        

    def forward(self, ids, masks):

        

        x = self.base_model(ids, attention_mask=masks)[1]

        x = self.fc1(x)

        return x

        
model = Model()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
def bert_encode(text, max_len=512):

    

    text = tokenizer.tokenize(text)

    text = text[:max_len-2]

    input_sequence = ["[CLS]"] + text + ["[SEP]"]

    tokens = tokenizer.convert_tokens_to_ids(input_sequence)

    tokens += [0] * (max_len - len(input_sequence))

    pad_masks = [1] * len(input_sequence) + [0] * (max_len - len(input_sequence))



    return tokens, pad_masks
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-z0-9] \n', '', text)

    return text
train_text = train.text[:6000]

val_text = train.text[6000:]
train_text = train_text.apply(clean_text)

val_text = val_text.apply(clean_text)
train_tokens = []

train_pad_masks = []

for text in train_text:

    tokens, masks = bert_encode(text)

    train_tokens.append(tokens)

    train_pad_masks.append(masks)

    

train_tokens = np.array(train_tokens)

train_pad_masks = np.array(train_pad_masks)
val_tokens = []

val_pad_masks = []

for text in val_text:

    tokens, masks = bert_encode(text)

    val_tokens.append(tokens)

    val_pad_masks.append(masks)

    

val_tokens = np.array(val_tokens)

val_pad_masks = np.array(val_pad_masks)


class Dataset(torch.utils.data.Dataset):

    

    def __init__(self, train_tokens, train_pad_masks, targets):

        

        super(Dataset, self).__init__()

        self.train_tokens = train_tokens

        self.train_pad_masks = train_pad_masks

        self.targets = targets

        

    def __getitem__(self, index):

        

        tokens = self.train_tokens[index]

        masks = self.train_pad_masks[index]

        target = self.targets[index]

        

        return (tokens, masks), target

    

    def __len__(self,):

        

        return len(self.train_tokens)
train_dataset = Dataset(

                    train_tokens=train_tokens,

                    train_pad_masks=train_pad_masks,

                    targets=train.target[:6000]

)
batch_size = 6

EPOCHS = 2
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
criterion = torch.nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.00001)
model.train()

y_preds = []



for epoch in range(EPOCHS):

        for i, ((tokens, masks), target) in enumerate(train_dataloader):



            y_pred = model(

                        tokens.long().to(device), 

                        masks.long().to(device)

                    )

            loss = criterion(y_pred, target[:, None].float().to(device))

            opt.zero_grad()

            loss.backward()

            opt.step()

            print('\rEpoch: %d/%d, %f%% loss: %0.2f'% (epoch+1, EPOCHS, i/len(train_dataloader)*100, loss.item()), end='')

        print()
val_dataset = Dataset(

                    train_tokens=val_tokens,

                    train_pad_masks=val_pad_masks,

                    targets=train.target[6000:].reset_index(drop=True)

)
val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=3, shuffle=False)
def accuracy(y_actual, y_pred):

    y_ = y_pred > 0

    return np.sum(y_actual == y_).astype('int') / y_actual.shape[0]
model.eval()

avg_acc = 0

for i, ((tokens, masks), target) in enumerate(val_dataloader):



    y_pred = model(

                tokens.long().to(device), 

                masks.long().to(device), 

            )

    loss = criterion(y_pred,  target[:, None].float().to(device))

    acc = accuracy(target.cpu().numpy(), y_pred.detach().cpu().numpy().squeeze())

    avg_acc += acc

    print('\r%0.2f%% loss: %0.2f, accuracy %0.2f'% (i/len(val_dataloader)*100, loss.item(), acc), end='')

print('\nAverage accuracy: ', avg_acc / len(val_dataloader))
class TestDataset(torch.utils.data.Dataset):

    

    def __init__(self, test_tokens, test_pad_masks):

        

        super(TestDataset, self).__init__()

        self.test_tokens = test_tokens

        self.test_pad_masks = test_pad_masks

        

    def __getitem__(self, index):

        

        tokens = self.test_tokens[index]

        masks = self.test_pad_masks[index]

        

        return (tokens, masks)

    

    def __len__(self,):

        

        return len(self.test_tokens)
test_tokens = []

test_pad_masks = []

for text in test.text:

    tokens, masks = bert_encode(text)

    test_tokens.append(tokens)

    test_pad_masks.append(masks)

    

test_tokens = np.array(test_tokens)

test_pad_masks = np.array(test_pad_masks)
test_dataset = TestDataset(

    test_tokens=test_tokens,

    test_pad_masks=test_pad_masks

)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=3, shuffle=False)
model.eval()

y_preds = []

for (tokens, masks) in test_dataloader:



    y_pred = model(

                tokens.long().to(device), 

                masks.long().to(device), 

            )

    y_preds += y_pred.detach().cpu().numpy().squeeze().tolist()
submission_df = pd.read_csv(os.path.join(path_to_dataset, 'sample_submission.csv'))
submission_df['target'] = (np.array(y_preds) > 0).astype('int')
submission_df.target.value_counts()
submission_df
submission_df.to_csv('submission.csv', index=False)