!pip install ../input/sacremoses/sacremoses-master/ > /dev/null



import os

import sys

import glob

import torch



sys.path.insert(0, "../input/transformers/transformers-master/")

import transformers



import torch

from transformers import *
import sys

package_dir_a = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"

sys.path.insert(0, package_dir_a)



from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch


from transformers import *

import shutil

# Translate model from tensorflow to pytorch

WORK_DIR = "../working/"

BERT_MODEL_PATH ='../input/bert-roberta/'

# BERT_MODEL_PATH ='../input/roberta-large/'

convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(

    BERT_MODEL_PATH + 'bert_model.ckpt',

BERT_MODEL_PATH + 'bert_config.json',

WORK_DIR + 'pytorch_model.bin')



shutil.copyfile(BERT_MODEL_PATH + 'bert_config.json', WORK_DIR + 'config.json')



print('yes')
import pandas as pd

import numpy as np

import json

import re

import copy

import random

from tqdm import tqdm

import collections

from random import shuffle

import warnings

from sklearn.metrics import *

warnings.filterwarnings("ignore")

tqdm.pandas()



#import torch.utils.data as data

from torchvision import datasets, models, transforms

from transformers import *

from sklearn.utils import shuffle

from sklearn.metrics import f1_score

import random



!pip install pytorch-crf
! ls ../input/ccks42/
class Do_data():

    def __init__(self,doc_id,content,enents,key_word=None):

        self.doc_id=doc_id

        self.content=content

        self.enents=enents

        self.key_word=key_word

    def __str__(self):

        return self.__repr__()

    def __repr__(self):

        s = ""

        s += "doc_id: %s\n" % (str(self.doc_id))

        s += "content: %s\n" % (str(self.content))

        s += "enents: %s\n" % (str(self.enents))

        s += "key_word: %s\n" % (str(self.key_word))

        return s





def get_shama(path):

    shama=collections.defaultdict(list)

    with open(path) as f:

        for l in f:

            l = json.loads(l)

            event_list=[] #事件集合

            content=l['content']

            docid=l['doc_id']

            enents=[]

            for event in l['events']:

                typ=event['event_type']

                v=list(event.keys())

                v.remove('event_type')

                v.remove('event_id')

                shama[typ].extend(v)

                shama[typ]=list(set(shama[typ]))

    return shama



def load_data(path,is_test=False):

    with open(path) as f:

        D=[]

        for l in f:

            l = json.loads(l)

            event_list=[] #事件集合

            content=l['content']

            docid=l['doc_id']

            enents=[]

            if not is_test:

                for event in l['events']:

                    event_dict={}

                    roles=shama[event['event_type']]

                    for role in roles:

                        if role in event:

                            event_dict[role]=event[role]

                    enents.append([event['event_type'],event_dict])

            data=Do_data(docid,content,enents)

#             print(data)

            D.append(data)

    return D

data_path='../input/ccks42/'

shama=get_shama(data_path+'event_element_train_data_label.txt')

train_data=load_data(data_path+'event_element_train_data_label.txt')

test_data=load_data(data_path+'event_element_dev_data.txt',is_test=True)

print(len(train_data))

print(len(test_data))
shama_label_ids={}

shama_ids_label={}

i=0

for k,v in shama.items():

    shama_label_ids[k]=i

    i+=1

for k,v in shama_label_ids.items():

    shama_ids_label[v]=k

shama_label_ids

shama_ids_label


MAX_LEN=512

SEP_TOKEN_ID = 102

DEVICE = 'cuda'



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





SEED=2020

seed_everything(SEED)
import unicodedata

def _is_control(ch):

    """控制类字符判断

    """

    return unicodedata.category(ch) in ('Cc', 'Cf')



def _is_special(ch):

    """判断是不是有特殊含义的符号

    """

    return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')



def stem(token):

    """获取token的“词干”（如果是##开头，则自动去掉##）

    """

    if token[:2] == '##':

        return token[2:]

    else:

        return token

def rematch(text,tokens,_do_lower_case=True):

    '''

    返回的是token后标签与原始的映射

    '''

    normalized_text, char_mapping = '', []

    #规范化样本

    for i, ch in enumerate(text):

        if _do_lower_case:

            ch = unicodedata.normalize('NFD', ch)

            ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])

            ch = ch.lower()

        ch = ''.join([

                c for c in ch

                if not (ord(c) == 0 or ord(c) == 0xfffd or _is_control(c))

            ])

        normalized_text += ch

        char_mapping.extend([i] * len(ch))



    text, token_mapping, offset = normalized_text, [], 0

    for i,token in enumerate(tokens):

        if _is_special(token):

            token_mapping.append([offset])

            offset+=1

        else:

            token = stem(token)

            start = text[offset:].index(token) + offset

            end = start + len(token)

            token_mapping.append(char_mapping[start:end])

            offset = end



    return token_mapping

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None)

text='雀巢裁员4000人：时代抛弃你时，连招呼都不会打！'

tokens=tokenizer.tokenize(text)

print(tokens)

mapping=rematch(text,tokens)
def search(pattern, sequence):

    """从sequence中寻找子串pattern

    如果找到，返回第一个下标；否则返回-1。

    """

    n = len(pattern)

    for i in range(len(sequence)):

        if sequence[i:i + n] == pattern:

            return i

    return -1





def search_list(pattern, sequence):

    """从sequence中寻找子串pattern

    如果找到，返回第一个下标；否则返回-1。

    """

    n = len(pattern)

    ans=[]

    for i in range(len(sequence)):

        if sequence[i:i + n] == pattern:

            ans.append(i)

    return ans





class Feature(object):

    def __init__(self,item_id,or_text,mapping,mapping_off,token_ids,labels,enent_type,cls_labels):

        self.id=item_id

        self.or_text=or_text

        self.mapping=mapping

        self.mapping_off=mapping_off

        self.token_ids=token_ids

        self.labels=labels

        self.enent_type=enent_type

        self.cls_labels=cls_labels

        

    def __str__(self):

        return self.__repr__()



    def __repr__(self):

        s = ""

        s += "id: %s" % (str(self.id))

        s += "or_text: %s" % (str(self.or_text))

        s += "mapping: %s" % (str(self.mapping))

        s += "mapping_off: %s" % (str(self.mapping_off))   

        s += "token_ids: %s" % (str(self.token_ids))

        s += "labels: %s" % (str(self.labels))

        s += "enent_type: %s" % (str(self.enent_type))

        s += "cls_labels: %s" % (str(self.cls_labels))

        return s



class zy_DataSet(torch.utils.data.Dataset):

    def __init__(self, data_list,train_mode=False, val_mode=False,test_mode=False,labeled=True):

        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None)

        self.train_mode = train_mode

        self.val_mode=val_mode

        self.test_mode=test_mode

        self.labeled = labeled

        self.data_list=data_list

        if self.train_mode:

            self.features =self.get_train_features(self.data_list,val_mode)

        elif self.val_mode:

            self.features =self.get_test_features(self.data_list)

        elif self.test_mode:

            self.features =self.get_test_features(self.data_list)

        else:

            print('no features !!!')

            

    def get_feat_qa(self,q_text,text):

        pass

    

    def get_test_features(self,data_list,val_mode=False):

        features=[]

        for index in tqdm(range(len(data_list))):

            data=data_list[index]

            item_id=data.doc_id #id

            text=data.content #文本内容

            enent_type='null'

            if len(data.enents)>0:

                enent_type=data.enents[0][0]

            tokens=self.tokenizer.tokenize(text)[:MAX_LEN-2]

#             mapping=rematch(text,tokens)

#             mapping_off=1 #解码转移

            tokens=['[CLS]'] +tokens +['[SEP]']

            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            if len(token_ids) < MAX_LEN:

                token_ids += [0] * (MAX_LEN- len(token_ids))

#             labels=[0]*len(token_ids)

#             cls_labels=shama_label_ids[enent_type]

            

            feature=Feature(item_id=item_id,

                            or_text=text,

                            mapping=0,

                            mapping_off=0,

                            token_ids=token_ids,

                            labels=0,

                            enent_type=enent_type,

                            cls_labels=0)

            features.append(feature)

        return features

    

    def get_train_features(self,data_list,val_mode=False):

        features=[]

        for index in tqdm(range(len(data_list))):

            data=data_list[index]

            item_id=data.doc_id #id

            text=data.content #文本内容

            enent_type=data.enents[0][0]

            tokens=self.tokenizer.tokenize(text)[:MAX_LEN-2]

#             mapping=rematch(text,tokens)

#             mapping_off=1 #解码转移

            tokens=['[CLS]'] +tokens +['[SEP]']

            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            if len(token_ids) < MAX_LEN:

                token_ids += [0] * (MAX_LEN- len(token_ids))

            labels=[0]*len(token_ids)

            cls_labels=shama_label_ids[enent_type]

            

            feature=Feature(item_id=item_id,

                            or_text=text,

                            mapping=0,

                            mapping_off=0,

                            token_ids=token_ids,

                            labels=labels,

                            enent_type=enent_type,

                            cls_labels=cls_labels)

            features.append(feature)

        return features



                        

    

    def __len__(self):

        return len(self.features)

    def select_tokens(self, tokens, max_num):

        if len(tokens) <= max_num:

            return tokens

        return tokens[:max_num]

    def get_seg_ids(self, ids):

        seg_ids = torch.zeros_like(ids)

        seg_idx = 0

        for i, e in enumerate(ids):

            seg_ids[i] = seg_idx

            if e == SEP_TOKEN_ID:

                seg_idx += 1

        max_idx = torch.nonzero(seg_ids == seg_idx)

        seg_ids[max_idx] = 0

        return seg_ids

    def __getitem__(self,index):

        feature=self.features[index]

        token_ids=torch.tensor(feature.token_ids)

        seg_ids=self.get_seg_ids(token_ids)

        cls_labels=feature.cls_labels

#         labels=np.concatenate([np.array(l) for l in labels])

        cls_labels=torch.tensor(np.array(cls_labels).astype(np.float32)).long()

#         cls_labels=torch.tensor(np.array(feature.cls_labels).astype(np.float32))

        return token_ids,seg_ids,cls_labels

    

    def collate_fn(self, batch):

        token_ids = torch.stack([x[0] for x in batch])

        seg_ids = torch.stack([x[1] for x in batch])

        cls_labels=torch.stack([x[2] for x in batch])

#         cls_labels=torch.stack([x[3] for x in batch])

        return token_ids, seg_ids, cls_labels

    

    

def get_loader(df,batch_size=16,train_mode=False,val_mode=False,test_mode=False,train_val=True):

    ds_df = zy_DataSet(copy.deepcopy(df),train_mode=train_mode,val_mode=val_mode,test_mode=test_mode,labeled=train_val)

    loader = torch.utils.data.DataLoader(ds_df, batch_size=batch_size, shuffle=train_mode, num_workers=0, collate_fn=ds_df.collate_fn, drop_last=train_mode)

    loader.num = len(ds_df)

    

    return loader,ds_df.features



def debug_loader():

    loader,features=get_loader(train_dict,train_mode=True)

    for token_ids, seg_ids,labels in train_loader:

        print(token_ids)

        print(seg_ids)

        print(labels)

        break

    print(len(features))


train_loader,train_features=get_loader(train_data,train_mode=True)

len(train_features)
train_features[0]
# debug_loader()
from transformers import *

from torchcrf import CRF

import torch

import torch.nn as nn

import torch.nn.functional as F

import time

from tqdm import tqdm_notebook



from transformers import *

import torch

import torch.nn as nn

import torch.nn.functional as F

import collections

import time

from tqdm import tqdm_notebook




class zy_Model(nn.Module):

    def __init__(self, n_classes=3):

        super(zy_Model, self).__init__()

        self.model_name = 'zy_Model'

        dim=768

        self.bert_model = BertModel.from_pretrained("../working",cache_dir=None,output_hidden_states=True)

#         self.hidden_fc=nn.Sequential(nn.Linear(dim, 3),nn.ReLU(True))

#         self.crf=CRF(3,batch_first=True)

        self.cls_fc=nn.Linear(dim, len(shama_label_ids))

        

    def mask_mean(self,x,mask):

        mask_x=x*(mask.unsqueeze(-1))

        x_sum=torch.sum(mask_x,dim=1)

        re_x=torch.div(x_sum,torch.sum(mask,dim=1).unsqueeze(-1))

        return re_x

    def forward(self,ids,seg_ids,cls_labels,is_test=False):

        attention_mask = (ids > 0)

        last_seq,pooled_output,hidden_state=self.bert_model(input_ids=ids,token_type_ids=seg_ids,attention_mask=attention_mask)

        

#         crf_feat=self.hidden_fc(last_seq)

        

#         crf_cls_feat=self.mask_mean(crf_feat,attention_mask*1)

        cls_out=self.cls_fc(pooled_output)

        

        if not is_test:

            criten=nn.CrossEntropyLoss()

            cls_loss=criten(cls_out,cls_labels)

            return cls_loss

        else:

            return cls_out

def debug_label():

    loader,features=get_loader(train_data[:64],train_mode=True,batch_size=2)

    model=zy_Model(3)

    for token_ids, seg_ids,cls_labels in loader:

        print(token_ids.size())

        loss= model(token_ids, seg_ids,cls_labels,is_test=False)

        print(loss)

        y= model(token_ids, seg_ids,cls_labels,is_test=True)

        print(len(y))

#         print(y[0].size())

#         print(out)

        break
debug_label()
# model=zy_Model(num_labels)

# list(model.named_parameters())
def get_text_f1(pred_text,text):

    common_number=0

    for ch in pred_text:

        if ch in text:

            common_number+=1

    p_len=len(pred_text)

    t_len=len(text)

    P=common_number/p_len if p_len>0 else 0.

    R=common_number/t_len if t_len>0 else 0.

    

    return (2*P*R)/(P+R) if (P+R)>0 else 0.

    

def metric_fn(results,features):

    totol_number=0

    predict_number=0

    predict_score=0

    for index,feat in enumerate(features):

        true_dicts=feat.answers #真实的answers

        pred_dicts=results[index] #预测的answers

        totol_number+=len(set(list(true_dicts.values()))) #去掉重复的

        predict_number+=len(set(list(pred_dicts.values())))

        

        #为了多个表述，改为dict list

        true_dicts_list=collections.defaultdict(list)

        predict_dicts_list=collections.defaultdict(list)

        for answer,key in pred_dicts.items():

            predict_dicts_list[key].append(answer)

        for answer,key in true_dicts.items():

            true_dicts_list[key].append(answer)

        #计算论文f1，如果有多个论元，选择分数最高那一个

        for key,answer in predict_dicts_list.items():

            if key in true_dicts_list:

                true_list=true_dicts_list[key]

                ans_list=answer

                s=0.

                for t in true_list:

                    for a in ans_list:

                        s=max(s,get_text_f1(a,t))

                predict_score+=s

    P=predict_score/predict_number if predict_number>0 else 0.

    R=predict_score/totol_number if totol_number>0 else 0.

    

    f1=(2*P*R)/(P+R) if (P+R)>0 else 0.

    

    return f1,P,R

    

def metric_fns(results,label_results,results_crf2,label_results_crf2,log=False):

    totol_number=0

    predict_number=0

    predict_score=0

    for item_id,feat in label_results.items():

#         if log:

#             print('id:',item_id)

#             print('type_labels:',label_results[item_id])

#             print('type_predict:',results[item_id])

#             print('type_labes_crf2:',label_results_crf2[item_id])

#             print('type_predict_crf2:',results_crf2[item_id])  

#             print('*'*50)

        #feat p_feat是一个list

        p_feat=results[item_id]

        predict_number+=len(p_feat)

        totol_number+=len(feat)

        for word in feat:

            if word in p_feat:

                predict_score+=1

            

    print(predict_number)

    print(totol_number)

    P=predict_score/predict_number if predict_number>0 else 0.

    R=predict_score/totol_number if totol_number>0 else 0.

    

    f1=(2*P*R)/(P+R) if (P+R)>0 else 0.

    

    return f1,P,R

    

    

def metric_fn_word(results,label_results,log=False):

    totol_number=0

    predict_number=0

    predict_score=0

    for item_id,feat in label_results.items():

        if log:

            print('id:',item_id)

            print('type_labels:',label_results[item_id])

            print('type_predict:',results[item_id])

            print('*'*50)

        #feat p_feat是一个list

        p_feat=results[item_id]

        predict_number+=len(p_feat)

        totol_number+=len(feat)

        for word in feat:

            if word in p_feat:

                predict_score+=1

            

    print(predict_number)

    print(totol_number)

    P=predict_score/predict_number if predict_number>0 else 0.

    R=predict_score/totol_number if totol_number>0 else 0.

    

    f1=(2*P*R)/(P+R) if (P+R)>0 else 0.

    

    return f1,P,R



def is_same_crf(out,c2):

    word=out[0]

    index=out[1]

    for c in c2:

        pword=c[0]

        pindex=c[1]

        if word==pword and index==pindex:

            return True

#         elif word in pword or pword in word:

#             return (len(pword)-len(word))==(pindex-index)

    return False

        



def validation_fn(model,val_loader,val_features,is_test=False):

    model.eval()

    bar = tqdm_notebook(val_loader)

    decodes=[]

    cls_preds=[]

    for i,(ids,seg_ids,cls_labels) in enumerate(bar):

        preds= model(ids.cuda(DEVICE),seg_ids.cuda(DEVICE),cls_labels.cuda(DEVICE),is_test=True)

#         decodes.extend(deocde)

        cls_preds.append(preds.detach().cpu().numpy())

    cls_preds=np.concatenate(cls_preds)

#     results=collections.defaultdict(list) #预测的结果

#     label_results=collections.defaultdict(list) #真实的结果

    cls_results=[] #预测的结果

    cls_label_results=[] #真实的结果

    cls_results_dict={}

    for index in range(len(cls_preds)):

#         decode=decodes[index] #解码序列

        cls_pred=cls_preds[index]

        feat=val_features[index]

        item_id=feat.id

        text=feat.or_text

        off=feat.mapping_off

        mapping=feat.mapping #映射

        typ=feat.enent_type

        if typ in shama_label_ids:

            cls_label_results.append(shama_label_ids[typ])

        cls_results.append(np.argmax(cls_pred))

        cls_results_dict[item_id]=shama_ids_label[np.argmax(cls_pred)]

        if typ!=shama_ids_label[np.argmax(cls_pred)] and typ in shama_label_ids:

            print(item_id)

            print('real:',typ)

            print('predict:',shama_ids_label[np.argmax(cls_pred)])

            print('*'*50)

    if not is_test:

        res=f1_score(cls_label_results,cls_results,average='macro')

        print(res)

        return res,0.,0.

    else:

        return cls_results_dict

def train_model(model,train_loader,val_loader,val_features,accumulation_steps=2,early_stop_epochs=2,epochs=4,model_save_path='pytorch_zy_model_true.pkl'):  

    

    losses=[]

    ########梯度累计

    batch_size = accumulation_steps*32

    

    ########早停

    no_improve_epochs=0

    

    ########优化器 学习率

    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    

#     crf_p=[n for n, p in param_optimizer if str(n).find('crf')!=-1]

#     print(crf_p)

    optimizer_grouped_parameters = [

            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.8},

            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},

#             {'params': [p for n, p in param_optimizer if n in crf_p], 'lr': 1e-4,'weight_decay': 0.8},

            ]   

    

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)

    

    train_len=len(train_loader)

    

    best_vmetric=-np.inf

    tloss = []

    cls_65_loss=[]

    for epoch in range(1,epochs+1):

        model.train()

        bar = tqdm_notebook(train_loader)

        for i,(ids,seg_ids,cls_labels) in enumerate(bar):

            loss = model(ids.cuda(DEVICE),seg_ids.cuda(DEVICE),cls_labels.cuda(DEVICE),is_test=False)

            sloss=loss

            sloss.backward()

            tloss.append(loss.item())

            cls_65_loss.append(loss.item())

            if (i+1) % accumulation_steps == 0 or (i+1)==train_len:

                optimizer.step()

                optimizer.zero_grad()

            bar.set_postfix(tloss=np.array(tloss).mean(),cls_65_loss=np.array(cls_65_loss).mean())

        

        #val

        val_f1,val_p,val_recall=validation_fn(model,val_loader,val_features)

        losses.append( 'train_loss:%.5f, f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %

            (np.array(tloss).mean(),val_f1, val_p, val_recall, best_vmetric))

        print(losses[-1])

        if val_f1>=best_vmetric:

            torch.save(model.state_dict(),model_save_path)

            best_vmetric=val_f1

            no_improve_epochs=0

            print('improve save model!!!')

        else:

            no_improve_epochs+=1

        if no_improve_epochs==early_stop_epochs:

            print('no improve score !!! stop train !!!')

            break

    return losses
test_loader,test_features=get_loader(test_data,batch_size=8,test_mode=True,train_val=False) #测试集
from sklearn.model_selection import KFold

FOLD=5

kf = KFold(n_splits=FOLD, shuffle=True,random_state=2019)

log_losses=[]



train_recall=collections.defaultdict(list)

train_recall_word=collections.defaultdict(list)

train_recall_real=collections.defaultdict(list)

test_recall=collections.defaultdict(list) #测试集

test_recall_word=collections.defaultdict(list)



for i,(train_index , test_index) in enumerate(kf.split(train_data)):

    print(str(i+1),'*'*50)

#     if i!=1:

#         continue

    tra=[train_data[i] for i in train_index]

    valid=[train_data[i] for i in test_index]

    print(len(tra))

    print(len(valid))

    

    tra_loader,tra_features=get_loader(tra,batch_size=16,train_mode=True,train_val=True)

    valid_loader,valid_features=get_loader(valid,batch_size=8,val_mode=True,train_val=True)

    

    model=zy_Model(3).cuda(DEVICE)

    losses=train_model(model,tra_loader,valid_loader,valid_features,accumulation_steps=1,early_stop_epochs=2,epochs=20,model_save_path='cv_recall{}.pkl'.format(i+1))

    log_losses.extend(losses)

    #加载最好模型

    model_save_path='cv_recall{}.pkl'.format(i+1)

    model.load_state_dict(torch.load(model_save_path))

    

    

    val_preds=validation_fn(model,valid_loader,valid_features,is_test=True)

    

    import pickle

    with open('val_preds{}.pkl'.format(i+1),'wb') as f:

        pickle.dump(val_preds,f)

        

    test_preds=validation_fn(model,test_loader,test_features,is_test=True)

    with open('test_preds{}.pkl'.format(i+1),'wb') as f:

        pickle.dump(test_preds,f)

    

    print(str(i+1),'-'*50)

    log_losses.append('*'*50)

    torch.cuda.empty_cache()

#     break

#     if i+1>=3:

#         break
for l in log_losses:

    print(l)
val_preds=validation_fn(model,valid_loader,valid_features,is_test=True)
test_preds=validation_fn(model,test_loader,test_features,is_test=True)
s={}

for k,v in test_preds.items():

    if v not in s:

        s[v]=0

    s[v]+=1
s
# val_f1,val_p,val_recall
# val_result,val_word_result=validation_fn(model,val_loader,val_features,is_test=True)
# val_word_result
# for k,v in val_word_result.items():

#     if len(v)==0:

#         print(k)

#         print(val_result[k])

#         if len(val_result[k])!=0:

#             for typ in val_result[k]:

#                 word=typ.split('-')[1]

#                 val_word_result[k].append((typ,word,-1))
# import pickle

# with open('right_recall1_val_large.pkl','wb') as f:

# #     pickle.dump(val_word_result,f)
# for k,v in test_words_result.items():

#     if len(v)==0:

#         print(k)

#         print(test_result[k])

#         if len(test_result[k])!=0:

#             for typ in test_result[k]:

#                 word=typ.split('-')[1]

#                 test_words_result[k].append((typ,word,-1))
# test_words_result
# test_final_result=collections.defaultdict(list) 

# for id,type in test_result.items():

#     if len(type)==0:

#         print(id)

#         print(type)

#     words=test_words_result[id]

#     types=[]

#     for k,_ in type.items():

#         types.append(k)

#     test_final_result[id].append(types)

#     test_final_result[id].append(words)
# import pickle

# with open('right_recall1_test_large.pkl','wb') as f:

#     pickle.dump(test_words_result,f)