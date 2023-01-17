from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

import os

print(os.listdir("../input/"))

print(os.listdir("../input/nvidiaapex/repository/NVIDIA-apex-39e153a"))

# Installing Nvidia Apex

! pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ../input/nvidiaapex/repository/NVIDIA-apex-39e153a







import pandas as pd

import numpy as np

import datetime

import pkg_resources

import seaborn as sns

import time

import scipy.stats as stats

import gc

import re

import operator 

import sys

from sklearn import metrics

from sklearn import model_selection

import torch

import torch.nn as nn

import torch.utils.data

import torch.nn.functional as F

from nltk.stem import PorterStemmer

from sklearn.metrics import roc_auc_score

%load_ext autoreload

%autoreload 2

%matplotlib inline

from tqdm import tqdm, tqdm_notebook

import os

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

import warnings

warnings.filterwarnings(action='once')

import pickle

from apex import amp

import shutil

device=torch.device('cuda')



MAX_SEQUENCE_LENGTH = 220

SEED = 1234

EPOCHS = 1

Data_dir="../input/jigsaw-unintended-bias-in-toxicity-classification"

Input_dir = "../input"

WORK_DIR = "../working/"

num_to_load=1000000                         #Train size to match time limit

valid_size= 100000                          #Validation Size

TOXICITY_COLUMN = 'target'



package_dir_a = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"

sys.path.insert(0, package_dir_a)



from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch

from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification,BertAdam
# This is the Bert configuration file

from pytorch_pretrained_bert import BertConfig



output_model_file = '../input/kernel6433b88374/bert_pytorch.bin'

bert_config = BertConfig('../input/kernel6433b88374/bert_config.json')
print("Predicting BERT base model......")





def convert_lines(example, max_seq_length,tokenizer):

    max_seq_length -=2

    all_tokens = []

    longer = 0

    for text in tqdm_notebook(example):

        tokens_a = tokenizer.tokenize(text)

        if len(tokens_a)>max_seq_length:

            tokens_a = tokens_a[:max_seq_length]

            longer += 1

        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))

        all_tokens.append(one_token)

    print(longer)

    return np.array(all_tokens)

BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=True)

y_columns=['target']

batch_size = 32



test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

X_test = convert_lines(test_df["comment_text"].fillna("DUMMY_VALUE"), MAX_SEQUENCE_LENGTH, tokenizer)

test = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))



model = BertForSequenceClassification(bert_config,num_labels=len(y_columns))

model.load_state_dict(torch.load(output_model_file ))

model.to(device)



for param in model.parameters():

    param.requires_grad = False

model.eval()



# Predicting

test_preds = np.zeros((test_df.shape[0],1))

model_preds = np.zeros((len(X_test)))

test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

tk0 = tqdm(test_loader)

for i, (x_batch,) in enumerate(tk0):

        pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)

        model_preds[i * batch_size:(i + 1) * batch_size] = pred[:, 0].detach().cpu().squeeze().numpy()



test_preds[:,0] = torch.sigmoid(torch.tensor(model_preds)).numpy().ravel()



#del model

#gc.collect()



# Sub-model prediction

bert_submission = pd.DataFrame.from_dict({

'id': test_df['id'],

'prediction': test_preds.mean(axis=1)})

bert_submission.to_csv('submission.csv', index=False)