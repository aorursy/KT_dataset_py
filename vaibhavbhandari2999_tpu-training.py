#Code to setup pytorch in kaggle. Found in a comment

!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev
import os

import torch

import pandas as pd

from scipy import stats

import numpy as np



from collections import OrderedDict, namedtuple

import torch.nn as nn

from torch.optim import lr_scheduler

import joblib



import logging

import transformers

from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule#AdamW is an adaptive optimizer

import sys

from sklearn import metrics, model_selection



#Import torch_xla for running pytorch on a TPU

import torch_xla

import torch_xla.debug.metrics as met

import torch_xla.distributed.data_parallel as dp

import torch_xla.distributed.parallel_loader as pl

import torch_xla.utils.utils as xu

import torch_xla.core.xla_model as xm

import torch_xla.distributed.xla_multiprocessing as xmp

import torch_xla.test.test_utils as test_utils



#Bert base uncased is used here. Uncased means that the text has been lowercased and it is the base model(not large model) of BERT

#Transformers are pre built architectures for NLP by HuggingFace

#BERT stands for Bidirectional Encoder Representations from Transformers. It trains bidirectional representations of text from left and right sides simultaneously.

#BERT is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left.



class BERTBaseUncased(nn.Module):

    def __init__(self, bert_path):

        super(BERTBaseUncased, self).__init__()

        self.bert_path = bert_path#Taking the pretrained model from TRANSFORMERS and defining the path(which is in config file)

        self.bert = transformers.BertModel.from_pretrained(self.bert_path)

        self.bert_drop = nn.Dropout(0.3)#Has a dropout 0f 0.3, that is 30% of the input tensors are zeroed out

        self.out = nn.Linear(768 * 2, 1)#We get a vector of size 768*2, one 768 for mean pooling, one 768 for max pooling



    def forward(self,ids,mask,token_type_ids):

        o1, _ = self.bert(ids,attention_mask=mask,token_type_ids=token_type_ids)#Here the underscore represnts that we dont need the second ouotput in the forward propogation step

        apool = torch.mean(o1, 1)#Both of these will be vectors of size 768 as the out layer is a vector of size 768(self.out)

        mpool, _ = torch.max(o1, 1)

        cat = torch.cat((apool, mpool), 1)#We concatenate both the poolings with axis 1

        bo = self.bert_drop(cat)

        p2 = self.out(bo)

        return p2



class BERTDatasetTraining:

    def __init__(self, comment_text, targets, tokenizer, max_length):

        self.comment_text = comment_text

        self.tokenizer = tokenizer

        self.max_length = max_length

        self.targets = targets



    def __len__(self):

        return len(self.comment_text)



    def __getitem__(self, item):

        comment_text = str(self.comment_text[item])

        comment_text = " ".join(comment_text.split())



        inputs = self.tokenizer.encode_plus(#from Hugging Face Tokenizers that encodes first and second string, but here there is no second string, so its None

            comment_text,            

            None,

            add_special_tokens=True,

            max_length=self.max_length,

        )

        ids = inputs["input_ids"]

        token_type_ids = inputs["token_type_ids"]

        mask = inputs["attention_mask"]

        

        padding_length = self.max_length - len(ids)

        

        ids = ids + ([0] * padding_length)#We padd it on the right for BERT

        mask = mask + ([0] * padding_length)

        token_type_ids = token_type_ids + ([0] * padding_length)

        

        return {

            'ids': torch.tensor(ids, dtype=torch.long),

            'mask': torch.tensor(mask, dtype=torch.long),

            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),

            'targets': torch.tensor(self.targets[item], dtype=torch.float)

        }
mx = BERTBaseUncased(bert_path="../input/bert-base-multilingual-uncased/")#The BERT Base Uncased model loaded, after adding it to the kaggle input data

df_train1 = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv", usecols=["comment_text", "toxic"]).fillna("none")

df_train2 = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv", usecols=["comment_text", "toxic"]).fillna("none")

df_train_full = pd.concat([df_train1, df_train2], axis=0).reset_index(drop=True)

df_train = df_train_full.sample(frac=1).reset_index(drop=True).head(200000)



df_valid = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/validation.csv', usecols=["comment_text", "toxic"])



#df_train = pd.concat([df_train, df_valid], axis=0).reset_index(drop=True)

#df_train = df_train.sample(frac=1).reset_index(drop=True)
def _run():

    def loss_fn(outputs, targets):

        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))#Simple loss function inbuilt



    def train_loop_fn(data_loader, model, optimizer, device, scheduler=None):

        model.train()

        #Go through all the batches inside Data Loader

        for bi, d in enumerate(data_loader):

            ids = d["ids"]

            mask = d["mask"]

            token_type_ids = d["token_type_ids"]

            targets = d["targets"]



            #Put all the values to the device you are using

            ids = ids.to(device, dtype=torch.long)

            mask = mask.to(device, dtype=torch.long)

            token_type_ids = token_type_ids.to(device, dtype=torch.long)

            targets = targets.to(device, dtype=torch.float)



            optimizer.zero_grad()#Clears the gradients of all optimized

            outputs = model(

                ids=ids,

                mask=mask,

                token_type_ids=token_type_ids

            )



            loss = loss_fn(outputs, targets)# Calculate batch loss based on CrossEntropy

            if bi % 10 == 0:

                xm.master_print(f'bi={bi}, loss={loss}')#Prints the loss for each batch index of multiples of 10



            loss.backward()#Back Propogation

            xm.optimizer_step(optimizer)#Optimizer function but should be called like this if using TPU

            if scheduler is not None:

                scheduler.step()#Needed to change the learning rate everytime



    def eval_loop_fn(data_loader, model, device):

        model.eval()

        fin_targets = []

        fin_outputs = []

        for bi, d in enumerate(data_loader):

            ids = d["ids"]

            mask = d["mask"]

            token_type_ids = d["token_type_ids"]

            targets = d["targets"]



            ids = ids.to(device, dtype=torch.long)

            mask = mask.to(device, dtype=torch.long)

            token_type_ids = token_type_ids.to(device, dtype=torch.long)

            targets = targets.to(device, dtype=torch.float)



            outputs = model(

                ids=ids,

                mask=mask,

                token_type_ids=token_type_ids

            )



            targets_np = targets.cpu().detach().numpy().tolist()

            outputs_np = outputs.cpu().detach().numpy().tolist()

            fin_targets.extend(targets_np)#Adds element to the list and extends it

            fin_outputs.extend(outputs_np)    



        return fin_outputs, fin_targets



    

    MAX_LEN = 192

    TRAIN_BATCH_SIZE = 64

    EPOCHS = 2



    tokenizer = transformers.BertTokenizer.from_pretrained("../input/bert-base-multilingual-uncased/", do_lower_case=True)



    train_targets = df_train.toxic.values

    valid_targets = df_valid.toxic.values



    train_dataset = BERTDatasetTraining(#calling Constructor of class BERTDatasetTraining 

        comment_text=df_train.comment_text.values,

        targets=train_targets,

        tokenizer=tokenizer,

        max_length=MAX_LEN

    )



    train_sampler = torch.utils.data.distributed.DistributedSampler(#We have to use distributedSampler for TPUs which distributes the data over multiple cores

          train_dataset,

          num_replicas=xm.xrt_world_size(),#World size if the number of cores being used

          rank=xm.get_ordinal(),#Its the rank of current process in num_replicas.Retrieves the replication ordinal of the current process.The ordinals range from 0 to xrt_world_size() minus 1.

          shuffle=True)#If true (default), sampler will shuffle the indices



    train_data_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=TRAIN_BATCH_SIZE,

        sampler=train_sampler,

        drop_last=True, #the drop_last argument drops the last non-full batch of each worker’s dataset replica.As each core creates a dataset replica for itself, if there are no equal batch sizes, it will crash

        num_workers=1

    )

#All validation functions are similar to training functions

    valid_dataset = BERTDatasetTraining(

        comment_text=df_valid.comment_text.values,

        targets=valid_targets,

        tokenizer=tokenizer,

        max_length=MAX_LEN

    )



    valid_sampler = torch.utils.data.distributed.DistributedSampler(

          valid_dataset,

          num_replicas=xm.xrt_world_size(),

          rank=xm.get_ordinal(),

          shuffle=False)



    valid_data_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=16,

        sampler=valid_sampler,

        drop_last=False,

        num_workers=1

    )



    #device = torch.device("cuda")

    #device = torch.device("cpu")

    device = xm.xla_device()#To recognize device as TPU

    model = mx.to(device)



    param_optimizer = list(model.named_parameters())# Get the list of named parameters

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']# Specify parameters where weight decay shouldn't be applied

    optimizer_grouped_parameters = [# Define two sets of parameters: those with weight decay, and those without

        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},

        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]



    lr = 3e-5 * xm.xrt_world_size()#Learning Rate

    #lr = 0.4 * 1e-5 * xm.xrt_world_size()#Learning Rate

    num_train_steps = int(len(train_dataset) / TRAIN_BATCH_SIZE / xm.xrt_world_size() * EPOCHS)#Steps will be divided by batch size and the number of cores being used

    xm.master_print(f'num_train_steps = {num_train_steps}, world_size={xm.xrt_world_size()}')



    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)# Instantiate AdamW optimizer with our two sets of parameters, and a learning rate

    scheduler = get_linear_schedule_with_warmup(#Create a schedule with a learning rate that decreases linearly after linearly increasing during a warmup period.

        optimizer,

        num_warmup_steps=0,

        num_training_steps=num_train_steps

    )



    for epoch in range(EPOCHS):

        para_loader = pl.ParallelLoader(train_data_loader, [device])#ParallelLoader loads the training data onto each device

        train_loop_fn(para_loader.per_device_loader(device), model, optimizer, device, scheduler=scheduler)#Call the train loop function



        para_loader = pl.ParallelLoader(valid_data_loader, [device])#Do same for validation data

        o, t = eval_loop_fn(para_loader.per_device_loader(device), model, device)

        xm.save(model.state_dict(), "model.bin")#Saves the file as model.bin in output files

        auc = metrics.roc_auc_score(np.array(t) >= 0.5, o)#Prints the AUC score as that is the one used for the competition

        xm.master_print(f'AUC = {auc}')

# Start training processes

def _mp_fn(rank, flags):

    torch.set_default_tensor_type('torch.FloatTensor')

    a = _run()



FLAGS={}

xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')