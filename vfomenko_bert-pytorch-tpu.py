!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev
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
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score



import torch

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW



from tqdm.notebook import tqdm
import torch_xla

import torch_xla.core.xla_model as xm



device = xm.xla_device()
tokenizer = BertTokenizerFast.from_pretrained('DeepPavlov/rubert-base-cased-conversational')
train = pd.read_csv('/kaggle/input/ml-guild-classification-task/train.csv')
texts = train['title'].fillna('') + train['review']
labels = ['positive', 'negative', 'neutral']

X_train, X_val, y_train, y_val = train_test_split(texts, train[labels].to_numpy().argmax(axis=1), random_state=0)
maxl = 512

batch_size = 32
X_train = torch.tensor([tokenizer.encode(text, max_length=maxl, pad_to_max_length=True, truncation=True) for text in tqdm(X_train)])

y_train = torch.tensor(y_train)

train_data = TensorDataset(X_train, y_train)

train_dataloader = DataLoader(

    train_data,

    sampler=RandomSampler(train_data),

    batch_size=batch_size,

    num_workers=4,

    pin_memory=True

)
X_val = torch.tensor([tokenizer.encode(text, max_length=maxl, pad_to_max_length=True, truncation=True) for text in tqdm(X_val)])

y_val = torch.tensor(y_val)

validation_data = TensorDataset(X_val, y_val)

validation_dataloader = DataLoader(

    validation_data,

    sampler=SequentialSampler(validation_data),

    batch_size=batch_size,

    num_workers=4,

    pin_memory=True

)
model = BertForSequenceClassification.from_pretrained('DeepPavlov/rubert-base-cased-conversational', num_labels=3)

model.to(device)
param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [

    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],

     'weight_decay_rate': 0.01},

    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],

     'weight_decay_rate': 0.0}

]



optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
for _ in range(2):

    model.train()

    train_loss = 0

    

    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_labels = batch

      

        optimizer.zero_grad()

      

        loss = model(b_input_ids.long(), token_type_ids=None, labels=b_labels)

        loss[0].backward()

        

        #optimizer.step()

        xm.optimizer_step(optimizer, barrier=True)

        

        train_loss += loss[0].item()

      

    print("Loss на обучающей выборке: {0:.5f}".format(train_loss / len(train_dataloader)))

    

    model.eval()



    valid_preds, valid_labels = [], []



    for batch in tqdm(validation_dataloader): 

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_labels = batch



        with torch.no_grad():

            logits = model(b_input_ids.long(), token_type_ids=None)



        logits = logits[0].detach().cpu()

        label_ids = b_labels.to('cpu').numpy()



        batch_preds = torch.softmax(logits, axis=1).numpy()

        batch_labels = label_ids

        valid_preds.extend(batch_preds)

        valid_labels.extend(batch_labels)



    valid_preds = np.array(valid_preds)

    print("roc-auc: " + str(roc_auc_score(valid_labels, valid_preds, multi_class='ovr')))
test = pd.read_csv('/kaggle/input/ml-guild-classification-task/test.csv')
test_texts = test['title'].fillna('') + test['review']
X_test = torch.tensor([tokenizer.encode(text, max_length=maxl, pad_to_max_length=True, truncation=True) for text in tqdm(test_texts)])

test_data = TensorDataset(X_test)

test_dataloader = DataLoader(

    test_data,

    sampler=SequentialSampler(test_data),

    batch_size=batch_size,

    num_workers=4,

    pin_memory=True

)
test_preds = []



for batch in tqdm(test_dataloader):   

    batch = batch[0]

    b_input_ids = batch.to(device)

    

    with torch.no_grad():

        logits = model(b_input_ids.long(), token_type_ids=None)



    logits = logits[0].detach().cpu()

    

    batch_preds = torch.softmax(logits, axis=1).numpy()

    test_preds.extend(batch_preds)

    

test_preds = np.array(test_preds)
sample_submission = pd.read_csv('/kaggle/input/ml-guild-classification-task/sample_submission.csv')

sample_submission[labels] = test_preds

sample_submission.to_csv('submission.csv', index=None)