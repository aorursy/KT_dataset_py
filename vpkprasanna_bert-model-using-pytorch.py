# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import torch

import pandas as pd

from scipy import stats

import numpy as np

import pandas as pd

from tqdm import tqdm

from collections import OrderedDict, namedtuple

import torch.nn as nn

from transformers import AdamW

from transformers import get_linear_schedule_with_warmup

from torch.optim import lr_scheduler

import joblib



import logging

import transformers

import sys

from sklearn import model_selection
class config:

    MAX_LEN = 512

    TRAIN_BATCH_SIZE = 64

    VALID_BATCH_SIZE = 16

    EPOCHS = 5

    BERT_PATH = "/kaggle/input/bert-base-uncased"

    MODEL_PATH = "model.bin"

    TRAINING_FILE = "/kaggle/input/nlp-getting-started/train.csv"

    TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH,lowercase=True)
class BERTDataset:

    def __init__(self, review, target):

        self.review = review

        self.target = target

        self.max_length = config.MAX_LEN

        self.tokenizer = config.TOKENIZER



    def __len__(self):

        return len(self.review)



    def __getitem__(self, item):

        review = str(self.review[item])

        review = " ".join(review.split())



        inputs = self.tokenizer.encode_plus(

            review,

            None,

            add_special_tokens=True,

            max_length=self.max_length,

        )

        ids = inputs["input_ids"]

        token_type_ids = inputs["token_type_ids"]

        mask = inputs["attention_mask"]

        

        padding_length = self.max_length - len(ids)

        

        ids = ids + ([0] * padding_length)

        mask = mask + ([0] * padding_length)

        token_type_ids = token_type_ids + ([0] * padding_length)

        

        return {

            'ids': torch.tensor(ids, dtype=torch.long),

            'mask': torch.tensor(mask, dtype=torch.long),

            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),

            'targets':torch.tensor(self.target[item],dtype=torch.float)



        }

class BERTBaseUncased(nn.Module):

    def __init__(self):

        super(BERTBaseUncased, self).__init__()

        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)

        self.bert_drop = nn.Dropout(0.3)

        self.out = nn.Linear(768, 1)



    def forward(self,ids,mask,token_type_ids):

        _, o2 = self.bert(

            ids,

            attention_mask=mask,

            token_type_ids=token_type_ids)

        

        bo = self.bert_drop(o2)

        output = self.out(bo)

        return output

def loss_fn(outpus,targets):

    return nn.BCEWithLogitsLoss()(outpus,targets.view(-1,1))
def train_fn(data_loader, model, optimizer, device, scheduler):

    model.train()



    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):

        ids = d["ids"]

        token_type_ids = d["token_type_ids"]

        mask = d["mask"]

        targets = d["targets"]



        ids = ids.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        targets = targets.to(device, dtype=torch.float)



        optimizer.zero_grad()

        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)



        loss = loss_fn(outputs, targets)

        loss.backward()

        optimizer.step()

        scheduler.step()

        
def eval_fn(data_loader,model,device):

    model.eval()

    fin_targets = []

    fin_outputs = []

    # for batch_index,dataset

    with torch.no_grad:

        for bi,d in tqdm(enumerate(data_loader),total = len(data_loader)):

            ids = d["ids"]

            toke_type_ids = d["token_type_ids"]

            mask = d["mask"]

            targets =d["targets"]



            ids = ids.to(device,dtype=torch.long)

            toke_type_ids = toke_type_ids.to(device,dtype=torch.long)

            mask = mask.to(device,dtype=torch.long)

            targets = targets.to(device,dtype=torch.float)



            outputs = model(

                ids= ids,

                mask = mask,

                token_type_ids = token_type_ids

            )

            fin_targets.extend(targets.cpu().detach().numpy().tolist())

            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

        

        return fin_outputs,fin_targets

dfx = pd.read_csv(config.TRAINING_FILE).fillna("none")
dfx.head()
def run(data_frame):

    df_train,df_valid = model_selection.train_test_split(data_frame,test_size=0.1,random_state=42,stratify=data_frame.target.values)

    

    df_train = df_train.reset_index(drop=True)

    df_valid = df_valid.reset_index(drop=True)

    

    train_dataset = BERTDataset(review=df_train.text.values,target=df_train.target.values)

    train_data_loader = torch.utils.data.DataLoader(train_dataset,batch_size=config.TRAIN_BATCH_SIZE,num_workers=4)

    

    

    valid_dataset = BERTDataset(review=df_valid.text.values,target=df_valid.target.values)

    valid_data_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=config.VALID_BATCH_SIZE,num_workers=2)

    

    device = torch.device("cuda")

    model = BERTBaseUncased()

    model.to(device)

    

    

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [

        {

            "params": [

                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],

            "weight_decay": 0.001},

        {

            "params": [

                p for n, p in param_optimizer if any(nd in n for nd in no_decay)],

            "weight_decay": 0.0}]

    

    

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

    optimizer = AdamW(optimizer_parameters, lr=3e-5)

    scheduler = get_linear_schedule_with_warmup(

        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps

    )



#     model = nn.DataParallel(model)



    best_accuracy = 0

    for epoch in range(config.EPOCHS):

        train_fn(train_data_loader, model, optimizer, device, scheduler)

        outputs, targets = engine.eval_fn(valid_data_loader, model, device)

        outputs = np.array(outputs) >= 0.5

        accuracy = metrics.accuracy_score(targets, outputs)

        print(f"Accuracy Score = {accuracy}")

        if accuracy > best_accuracy:

            torch.save(model.state_dict(), config.MODEL_PATH)

            best_accuracy = accuracy



    
run(data_frame=dfx)