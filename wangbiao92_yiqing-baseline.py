# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from pytorch_transformers import  BertModel, BertConfig, BertTokenizer, BertForSequenceClassification, AdamW

from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import f1_score

import torch

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset

from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm, trange

import pandas as pd

import numpy as np

import time 

# import torch_xla

# import torch_xla.core.xla_model as xm
import random

def seed_everything(seed):

    """

    Seeds basic parameters for reproductibility of results

    

    Arguments:

        seed {int} -- Number of the seed

    """

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

seed = 2020

seed_everything(seed)

start_time = time.time()
# 编码转utf8

def re_encode(path):

    with open(path, 'r', encoding='GB2312', errors='ignore') as file:

        lines = file.readlines()

    with open('/kaggle/working/nCov_10k_test.csv', 'w', encoding='utf-8') as file:

        file.write(''.join(lines))

        file.close()

        

re_encode('/kaggle/input/yiqing/nCov_10k_test.csv')



def re_encode(path):

    with open(path, 'r', encoding='GB2312', errors='ignore') as file:

        lines = file.readlines()

    with open('/kaggle/working/nCoV_100k_train.labled.csv', 'w', encoding='utf-8') as file:

        file.write(''.join(lines))

        file.close()

re_encode('/kaggle/input/yiqing/nCoV_100k_train.labled.csv')


bert_pre_model='/kaggle/input/yiqing/pytorch_model.bin'#预训练模型文件

bert_config='/kaggle/input/yiqing/config.json'#配置文件

bert_pre_tokenizer='/kaggle/input/yiqing/vocab.txt'#词表



# 加载训练集、测试集

train_df = pd.read_csv('/kaggle/working/nCoV_100k_train.labled.csv',engine ='python')

test_df  = pd.read_csv('/kaggle/working/nCov_10k_test.csv',engine ='python')



train_df = train_df[train_df['情感倾向'].isin(['0','1','-1'])]

train_df.columns=["id", "time", "con", "txt", "pic", "video", "label"]

train_df["label"] = train_df["label"].astype("int")



# test_df = test_df[test_df['情感倾向'].isin(['0','1','-1'])]

test_df.columns=["id", "time", "con", "txt", "pic", "video"]

# test_df["label"] = test_df["label"].astype("int")





df = train_df.loc[:2000,:].copy()

df_dev = test_df.loc[:1000,:].copy()



#提取语句并处理

sentencses=['[CLS] ' + str(sent) + ' [SEP]' for sent in df.txt.values]

labels=df.label.values
#这里中性还是使用0表示1表示积极2表示不积极

tokenizer=BertTokenizer.from_pretrained(bert_pre_tokenizer,do_lower_case=True)



labels=list(map(lambda x:0 if x == 0 else 1 if x == 1 else 2,[x for x in labels]))

tokenized_sents=[tokenizer.tokenize(sent) for sent in sentencses]

print("tokenized的第一句话:",tokenized_sents[0])



MAX_LEN=256

#训练集部分

#将分割后的句子转化成数字  word-->idx

input_ids=[tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents]

print("转化后的第一个句子:",input_ids[0])

#做PADDING

#大于256做截断，小于256做PADDING

input_ids=pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

print("Padding 第一个句子:",input_ids[0])

#建立mask

attention_masks = []

for seq in input_ids:

  seq_mask = [float(i>0) for i in seq]

  attention_masks.append(seq_mask)

print("第一个attention mask:",attention_masks[0])



#测试集部分

#构建测试集

dev_sentencses=['[CLS] ' + str(sent) + ' [SEP]' for sent in df_dev.txt.values]

# dev_labels=df_dev.label.values

# print("dev_label:",dev_labels[100:110])

# dev_labels=list(map(lambda x:0 if x == 0 else 1 if x == 1 else 2,[x for x in dev_labels]))

# dev_labels=[to_categorical(i, num_classes=3) for i in dev_labels]

dev_tokenized_sents=[tokenizer.tokenize(sent) for sent in dev_sentencses]

dev_input_ids=[tokenizer.convert_tokens_to_ids(sent) for sent in dev_tokenized_sents]

dev_input_ids=pad_sequences(dev_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

dev_attention_masks = []

for seq in dev_input_ids:

  dev_seq_mask = [float(i>0) for i in seq]

  dev_attention_masks.append(dev_seq_mask)
# 构建训练集和测试集的dataloader

train_inputs = torch.tensor(input_ids)

validation_inputs = torch.tensor(dev_input_ids)

train_labels = torch.tensor(labels)

# validation_labels = torch.tensor(dev_labels)

train_masks = torch.tensor(attention_masks)

validation_masks = torch.tensor(dev_attention_masks)



batch_size = 32

train_data = TensorDataset(train_inputs, train_masks, train_labels)

train_sampler = RandomSampler(train_data)

train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)

validation_data = TensorDataset(validation_inputs, validation_masks)



validation_sampler = SequentialSampler(validation_data)

validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)





#装载预训练bert模型

modelConfig = BertConfig.from_pretrained(bert_config, num_labels=3)

model = BertForSequenceClassification.from_pretrained(bert_pre_model, config=modelConfig)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = xm.xla_device()

model = model.to(device)

# device = "cpu"

# print(model.cuda())

# 优化器

param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'gamma', 'beta']

optimizer_grouped_parameters = [

    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],

     'weight_decay_rate': 0.01},

    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],

     'weight_decay_rate': 0.0}

]



# optimizer = AdamW(optimizer_grouped_parameters,

#                      lr=2e-5,

#                      warmup=.1)



optimizer = AdamW(optimizer_grouped_parameters,

                     lr=2e-5)



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#定义一个计算准确率的函数 用f1-score

# def flat_accuracy(preds, labels):

#     pred_flat = np.argmax(preds, axis=1).flatten()

#     labels_flat = labels.flatten()

#     return np.sum(pred_flat == labels_flat) / len(labels_flat)



#训练开始

train_loss_set = []#可以将loss加入到列表中，后期画图使用

epochs = 1

for _ in trange(epochs, desc="Epoch"):

    #训练开始

    model.train()

    tr_loss = 0

    nb_tr_examples, nb_tr_steps = 0, 0

    for step, batch in enumerate(train_dataloader):

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        b_input_ids = b_input_ids.to(torch.int64)

#         b_labels = b_labels.unsqueeze(1)

        optimizer.zero_grad()

        #取第一个位置，BertForSequenceClassification第一个位置是Loss，第二个位置是[CLS]的logits

#         b_input_ids = b_input_ids.to(device)

#         b_input_mask = b_input_mask.to(device)

#         b_labels = b_labels.to(device)

        loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)[0]

        train_loss_set.append(loss.item())

        loss.backward()

        optimizer.step()



        tr_loss += loss.item()

        nb_tr_examples += b_input_ids.size(0)

        nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss / nb_tr_steps))

    #模型评估

    model.eval()

    eval_loss, eval_accuracy = 0, 0

    nb_eval_steps, nb_eval_examples = 0, 0

    pre_lst = np.array([], dtype=int)

    test_lst = np.array([], dtype=int)

    for batch in validation_dataloader:

        batch = tuple(t.to(device) for t in batch)

#         b_input_ids, b_input_mask, b_labels = batch

        b_input_ids, b_input_mask = batch

#         b_labels = b_labels.unsqueeze(1)

        with torch.no_grad():

            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]

        logits = logits.detach().cpu().numpy()

#         label_ids = b_labels.to('cpu').numpy()

        

        pre = np.argmax(logits, axis=1).flatten()

        pre_lst = np.concatenate((pre_lst, pre), axis=0)

#         test_lst = np.concatenate((test_lst, label_ids), axis=0)

#         tmp_eval_accuracy = flat_accuracy(logits, label_ids)

#         eval_accuracy += tmp_eval_accuracy

#         nb_eval_steps += 1

#     print("F1_score: {}".format(f1_score(test_lst,pre_lst,labels=[0,1,2],average='micro')))

end_time = time.time()

print("spend time:{}".format(end_time-start_time))

submission = test_df[["id"]].copy()

submission["y"] = pre_lst

submission.loc[submission["y"] == 2, "y"] = -1

submission.to_csv("/kaggle/working/submission.csv", index=False, encoding="utf8")