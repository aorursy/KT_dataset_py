

import re

import numpy as np

import pandas as pd

import transformers

import tokenizers

import torch

import torch.nn as nn

from tqdm.autonotebook import tqdm

from transformers import get_linear_schedule_with_warmup

from transformers import AdamW

MAX_LEN=192

EPOCHS=5

TRAIN_BATCH_SIZE=64

VALID_BATCH_SIZE=16

BERT_PATH="../input/roberta-base"

MODEL_PATH = "model.bin"

TOKENIZER=tokenizers.ByteLevelBPETokenizer(

    vocab_file=f'{BERT_PATH}/vocab.json',

    merges_file=f'{BERT_PATH}/merges.txt',

    lowercase=True,

    add_prefix_space=True

)

def process_data(tweet, selected_text, sentiment, max_len, tokenizer):

    selected_text=' '.join(str(selected_text).split())

    tweet= " ".join(str(tweet).split())

    len_st = len(str(selected_text))

    idx0 = None

    idx1 = None

    for ind in (i for i, e in enumerate(str(tweet)) if e == selected_text[0]):

        if tweet[ind: ind+len_st] == selected_text:

            idx0 = ind

            idx1 = ind + len_st - 1

            break



    char_targets = [0] * len(tweet)

    if idx0 != None and idx1 != None:

        for ct in range(idx0, idx1 + 1):

            char_targets[ct] = 1

    

    tok_tweet = tokenizer.encode(tweet)

    ids_orig = tok_tweet.ids

    tweet_offset = tok_tweet.offsets

    

    target_idx = []

    for j, (offset1, offset2) in enumerate(tweet_offset):

        if sum(char_targets[offset1: offset2]) > 0:

            target_idx.append(j)

    orig_text=tweet

    target_start = target_idx[0]

    target_end = target_idx[-1]



    sentiment_id = {

        'positive': 1313,

        'negative': 2430,

        'neutral': 7974

    }

    

    ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + ids_orig + [2]

    token_type_ids = [0, 0, 0, 0] + [0] * (len(ids_orig) + 1)

    mask = [1] * len(token_type_ids)

    tweet_offset = [(0, 0)] * 4 + tweet_offset + [(0, 0)]

    target_start += 4

    target_end += 4



    padding_length = max_len - len(ids)

    if padding_length > 0:

        ids = ids + ([0] * padding_length)

        mask = mask + ([0] * padding_length)

        token_type_ids = token_type_ids + ([0] * padding_length)

        tweet_offset = tweet_offset + ([(0, 0)] * padding_length)

    return {

        'ids': ids,

        'text_offset': tweet_offset,

        'token_type_ids': token_type_ids,

        'ids_orig': ids_orig,

        'masks': mask,

        'target_end': target_end,

        'target_start': target_start,

        'sentiment': sentiment,

        'selected_text': str(selected_text),

        'orig_text': orig_text 

    }

class BertData:

    def __init__(self,text,selected_text,sentiment):

        self.text = text

        self.selected_text = selected_text

        self.sentiment = sentiment

        self.max_len=MAX_LEN

        self.tokenizer=TOKENIZER

    

    def __len__(self):

        return len(self.text)



    def __getitem__(self,item):

        data = process_data(self.text[item],

                            self.selected_text[item],

                            self.sentiment[item],

                            self.max_len,

                            self.tokenizer)

        return {

        'ids': torch.tensor (data['ids'], dtype=torch.long),

        'text_offset': torch.tensor (data['text_offset'], dtype=torch.long),

        'token_type_ids': torch.tensor (data['token_type_ids'], dtype=torch.long),

        'masks': torch.tensor (data['masks'], dtype=torch.long),

        'target_end': torch.tensor (data['target_end'], dtype=torch.long),

        'target_start': torch.tensor (data['target_start'], dtype=torch.long),

        'sentiment': data['sentiment'],

        'selected_text': data['selected_text'],

        'orig_text': data['orig_text'] 

    }

class bertmodel(transformers.BertPreTrainedModel):

    def __init__(self, conf):

        super(bertmodel, self).__init__(conf)

        self.bert = transformers.RobertaModel.from_pretrained(BERT_PATH, config=conf)

        self.dropout= nn.Dropout(0.5)

        

        self.bn1 = nn.BatchNorm1d(num_features=192)

        self.bn2 = nn.BatchNorm1d(num_features=192)

        

        

        self.c1=nn.Conv1d(768,768,2)

        self.c11=nn.Conv1d(768,256,2)

        self.c111=nn.Conv1d(256,64,2)

        self.c2=nn.Conv1d(768,768,2)

        self.c22=nn.Conv1d(768,256,2)

        self.c222=nn.Conv1d(256,64,2)

        self.Leaky= nn.ReLU(0.3)

        self.i0=nn.Linear(64,1)

        self.i1=nn.Linear(64,1)

        nn.init.normal_(self.i0.bias, 0)

        nn.init.normal_(self.i0.weight, std=0.02)

        nn.init.normal_(self.i1.bias, 0)

        nn.init.normal_(self.i1.weight, std=0.02)

        



    def forward(self, ids, masks, token_type_ids):

        _,_,out=self.bert(

            ids,

            attention_mask=masks,

            token_type_ids=token_type_ids

        )

        out = torch.stack([out[-1], out[-2], out[-3], out[-4]])

        out = torch.mean(out, 0)



        #out=torch.cat((out[-1],out[-2]), dim=-1)

        out=self.dropout(out)

        out = nn.functional.pad(out.transpose(1,2), (1, 0))



        out1 = self.c1(out).transpose(1,2)

        out1=self.Leaky(self.bn1 (out1))

        out1 = self.c11(nn.functional.pad(out1.transpose(1,2), (1, 0))).transpose(1,2)

        out1=self.Leaky(self.bn2 (out1))

        out1 = self.c111(nn.functional.pad(out1.transpose(1,2), (1, 0))).transpose(1,2)

        out1=self.Leaky(self.bn2 (out1))

        

        out2 = self.c2(out).transpose(1,2)

        out2=self.Leaky(self.bn1 (out2))

        out2 = self.c22(nn.functional.pad(out2.transpose(1,2), (1, 0))).transpose(1,2)

        out2=self.Leaky(self.bn2 (out2))

        out2 = self.c222(nn.functional.pad(out2.transpose(1,2), (1, 0))).transpose(1,2)

        out2=self.Leaky(self.bn2 (out2))

        start_logits = self.i0(self.dropout(out1)).squeeze(-1)

        end_logits = self.i1(self.dropout(out2)).squeeze(-1)

        return start_logits, end_logits





def loss_fn(start_logits, end_logits, start_pos, end_pos):

    loss= nn.CrossEntropyLoss()

    start_loss=loss(start_logits, start_pos)

    end_loss=loss(end_logits, end_pos)

    total_loss=(start_loss+end_loss)

    return total_loss

def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))



def calculate_jaccard(orig_text, sentiment, target_str, ind_start, ind_end, offsets, verbose=False):

    filtered_out=''

    if ind_start>ind_end:

        filtered_out=orig_text

    

    for i in range(ind_start, ind_end+1):

        filtered_out+=orig_text[offsets[i][0]:offsets[i][1]]

        if (i+1)<len(offsets) and offsets[i][1]<offsets[i+1][0]:

            filtered_out+=' '

    if sentiment=='neutral' or len(str(orig_text).split())<2:

        filtered_out=orig_text

    

    jac=jaccard(str(target_str).strip(),str(filtered_out).strip())

    return jac, filtered_out

class AverageMeter:

    """Computes and stores the average and current value"""

    def __init__(self):

        self.reset()



    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count

def train_fn(data_loader, model, optimizer, device, scheduler=None):

    model.train()

    losses=AverageMeter()

    jaccards=AverageMeter()



    tk0 = tqdm(data_loader, total=len(data_loader))

    for ind, d in enumerate(tk0,0):

        

        ids = d['ids']

        text_offset = d['text_offset']

        token_type_ids = d['token_type_ids']

        target_end = d['target_end']

        target_start = d['target_start']

        sentiment = d['sentiment']

        selected_text = d['selected_text']

        masks = d['masks']

        orig_text = d['orig_text']

        

        ids = ids.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        target_start = target_start.to(device, dtype=torch.long)

        target_end = target_end.to(device, dtype=torch.long)

        masks = masks.to(device, dtype=torch.long)



        model.zero_grad()

        out_start, out_end = model(

            ids,

            token_type_ids = token_type_ids,

            masks = masks

        )



        loss=loss_fn(out_start, out_end, target_start, target_end)

        loss.backward()

        optimizer.step()

        scheduler.step()



        out_start = torch.softmax(out_start, dim=1).cpu().detach().numpy()

        out_end = torch.softmax(out_end, dim=1).cpu().detach().numpy()  



        jaccard_scores=[]

        for ind, x in enumerate(orig_text):

            selected=selected_text[ind]

            sentiment1=sentiment[ind]

            jaccard_score,_=calculate_jaccard(x,

                        sentiment1,

                        selected,

                        np.argmax(out_start[ind,:]),

                        np.argmax(out_end[ind,:]),

                        text_offset[ind]

            )

            jaccard_scores.append(jaccard_score)

        jaccards.update(np.mean(jaccard_scores),ids.size(0))

        losses.update(loss.item(),ids.size(0))

        tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)

def eval_fn(data_loader,model, device):

    model.eval()

    losses=AverageMeter()

    jaccards=AverageMeter()



    with torch.no_grad():

        tk0 = tqdm(data_loader, total=len(data_loader))

        for ind, d in enumerate(tk0,0):

        

            ids = d['ids']

            text_offset = d['text_offset'].numpy()

            token_type_ids = d['token_type_ids']

            target_end = d['target_end']

            target_start = d['target_start']

            sentiment = d['sentiment']

            selected_text = d['selected_text']

            masks = d['masks']

            orig_text = d['orig_text']

        

            ids = ids.to(device, dtype=torch.long)

            token_type_ids = token_type_ids.to(device, dtype=torch.long)

            target_start = target_start.to(device, dtype=torch.long)

            target_end = target_end.to(device, dtype=torch.long)

            masks = masks.to(device, dtype=torch.long)



            out_start, out_end = model(

            ids=ids,

            token_type_ids = token_type_ids,

            masks = masks

            )



            loss=loss_fn(out_start, out_end, target_start, target_end)

            

            out_start = torch.softmax(out_start, dim=1).cpu().detach().numpy()

            out_end = torch.softmax(out_end, dim=1).cpu().detach().numpy()  



            jaccard_scores=[]

            for ind, x in enumerate(orig_text):

                selected=selected_text[ind]

                sentiment1=sentiment[ind]

                jaccard_score,_=calculate_jaccard(x,

                        sentiment1,

                        selected,

                        np.argmax(out_start[ind,:]),

                        np.argmax(out_end[ind,:]),

                        text_offset[ind]



                )

                jaccard_scores.append(jaccard_score)

            jaccards.update(np.mean(jaccard_scores),ids.size(0))

            losses.update(loss.item(),ids.size(0))

            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)

    print(f'jaccard score: {jaccards.avg}')

    return jaccards.avg

class EarlyStopping:

    def __init__(self, patience=7, mode="max", delta=0.001):

        self.patience = patience

        self.counter = 0

        self.mode = mode

        self.best_score = None

        self.early_stop = False

        self.delta = delta

        if self.mode == "min":

            self.val_score = np.Inf

        else:

            self.val_score = -np.Inf



    def __call__(self, epoch_score, model, model_path):



        if self.mode == "min":

            score = -1.0 * epoch_score

        else:

            score = np.copy(epoch_score)



        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(epoch_score, model, model_path)

        elif score < self.best_score + self.delta:

            self.counter += 1

            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_score = score

            self.save_checkpoint(epoch_score, model, model_path)

            self.counter = 0



    def save_checkpoint(self, epoch_score, model, model_path):

        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:

            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))

            torch.save(model.state_dict(), model_path)

        self.val_score = epoch_score

def run(fold):

    df=pd.read_csv('../input/tweet-sentiment-extraction/train.csv')

    df_len_per_fold=int(len(df)/5)

    df_fold=df.copy()

    df_fold.loc[:,'kfold']=-1

    df_fold.loc[:df_len_per_fold*1,'kfold']=0

    df_fold.loc[df_len_per_fold*1:df_len_per_fold*2,'kfold']=1

    df_fold.loc[df_len_per_fold*2:df_len_per_fold*3,'kfold']=2

    df_fold.loc[df_len_per_fold*3:df_len_per_fold*4,'kfold']=3

    df_fold.loc[df_len_per_fold*4:,'kfold']=4

    

    train_data_fold= df_fold[df_fold.loc[:,'kfold']!=fold].reset_index(drop=True)

    valid_data_fold= df_fold[df_fold.loc[:,'kfold']==fold].reset_index(drop=True)



    train_data=BertData(

        text=train_data_fold.text.values,

        selected_text=train_data_fold.selected_text.values,

        sentiment=train_data_fold.sentiment.values

    )



    train_data_loader= torch.utils.data.DataLoader(

        train_data,

        batch_size=TRAIN_BATCH_SIZE,

        num_workers=4

    )



    valid_data=BertData(

        text=valid_data_fold.text.values,

        selected_text=valid_data_fold.selected_text.values,

        sentiment=valid_data_fold.sentiment.values

    )



    valid_data_loader= torch.utils.data.DataLoader(

        valid_data,

        batch_size=VALID_BATCH_SIZE,

        num_workers=2

    )



    device = torch.device("cuda")

    model_config= transformers.BertConfig.from_pretrained(BERT_PATH)

    model_config.output_hidden_states=True

    model= bertmodel(conf=model_config)

    model.to(device)



    train_steps= len(train_data_fold)/TRAIN_BATCH_SIZE* EPOCHS

    param_optimizer= list(model.named_parameters())

    no_decay=['bias', 'LayerNorm.bias', 'LayerNorm.weights']

    optimizer_paramters=[

                         {'params': [para for name, para in param_optimizer if not any(nd in name for nd in no_decay)], 'weight_decay': 0.001},

                         {'params': [para for name, para in param_optimizer if any(nd in name for nd in no_decay)], 'weight_decay': 0.0}

    ]



    optimizer= AdamW(optimizer_paramters, lr=3e-5)

    scheduler= get_linear_schedule_with_warmup(

        optimizer, 

        num_warmup_steps=0, 

        num_training_steps= train_steps

    )



    es= EarlyStopping(patience=6, mode='max')

    print(f'Training strated for fold {fold}')



    for epoch in range(5):

        train_fn(train_data_loader, model,optimizer ,device, scheduler=scheduler)

        jaccard=eval_fn(valid_data_loader, model, device)

        print(f'Jaccard score: {jaccard}')

        es(jaccard, model, model_path=f'model_{fold}.bin')

        if es.early_stop:

            print('Early stopping...')

            break



run(fold=0)
run(fold=1)
run(fold=2)
run(fold=3)
run(fold=4)
df_test=pd.read_csv('../input/tweet-sentiment-extraction/test.csv')

df_test.loc[:,'selected_text']=df_test.text.values

device='cuda'

model_config=transformers.BertConfig.from_pretrained(BERT_PATH)

model_config.output_hidden_states=True

model1=bertmodel(conf=model_config)

model1.to(device)

model1.load_state_dict(torch.load('model_0.bin'))

model1.eval()



model2=bertmodel(conf=model_config)

model2.to(device)

model2.load_state_dict(torch.load('model_1.bin'))

model2.eval()



model3=bertmodel(conf=model_config)

model3.to(device)

model3.load_state_dict(torch.load('model_2.bin'))

model3.eval()



model4=bertmodel(conf=model_config)

model4.to(device)

model4.load_state_dict(torch.load('model_3.bin'))

model4.eval()



model5=bertmodel(conf=model_config)

model5.to(device)

model5.load_state_dict(torch.load('model_4.bin'))

model5.eval()

final_out=[]



test_data=BertData(

    text=df_test.text.values,

    selected_text=df_test.selected_text.values,

    sentiment=df_test.sentiment.values

)



test_data_loader=torch.utils.data.DataLoader(

    test_data,

    batch_size=VALID_BATCH_SIZE,

    shuffle=False,

    num_workers=1

)



with torch.no_grad():

    tk0=tqdm(test_data_loader, total=len(test_data_loader))

    for ind, d in enumerate(tk0):

        ids = d['ids']

        text_offset = d['text_offset']

        token_type_ids = d['token_type_ids']

        target_end = d['target_end']

        target_start = d['target_start']

        sentiment = d['sentiment']

        selected_text = d['selected_text']

        masks = d['masks']

        orig_text = d['orig_text']

        

        ids = ids.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        target_start = target_start.to(device, dtype=torch.long)

        target_end = target_end.to(device, dtype=torch.long)

        masks = masks.to(device, dtype=torch.long)



        out_start1, out_end1 = model1(

            ids,

            token_type_ids = token_type_ids,

            masks = masks

            )

        out_start2, out_end2 = model2(

            ids,

            token_type_ids = token_type_ids,

            masks = masks

            )

        out_start3, out_end3 = model3(

            ids,

            token_type_ids = token_type_ids,

            masks = masks

            )

        out_start4, out_end4 = model4(

            ids,

            token_type_ids = token_type_ids,

            masks = masks

            )

        out_start5, out_end5 = model5(

            ids,

            token_type_ids = token_type_ids,

            masks = masks

            )

        out_start=(out_start1 + out_start2 + out_start3 + out_start4 + out_start5)/5

        out_end=(out_end1 + out_end2 + out_end3 + out_end4 + out_end5)/5



        out_start = torch.softmax(out_start, dim=1).cpu().detach().numpy()

        out_end = torch.softmax(out_end, dim=1).cpu().detach().numpy()



        for ind, x in enumerate(orig_text):

            selected=selected_text[ind]

            sentiment1=sentiment[ind]

            _, out_sequences=calculate_jaccard(

                       x,

                       sentiment1,

                        selected,

                        np.argmax(out_start[ind,:]),

                        np.argmax(out_end[ind,:]),

                        text_offset[ind]

            )

            final_out.append(out_sequences)



def process(selected):

    return ' '.join(set(selected.lower().split()))
sample=pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')

sample.loc[:,'selected_text']=final_out

sample.selected_text=sample.selected_text.map(process)

sample.to_csv('submission.csv',index=False)