# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm,trange,tqdm_notebook



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from torch import nn

import torch

from torch.utils.data import Dataset, DataLoader

import transformers

from transformers.optimization import AdamW

import tokenizers

from wordcloud import WordCloud, STOPWORDS

import seaborn as sns

import matplotlib.pyplot as plt
basePath = '/kaggle/input/tweet-sentiment-extraction/'

raw_train_data = pd.read_csv(basePath+'train.csv')

raw_train_data.head(15)
class config:

    BERT_PATH = "../input/bert-base-uncased/"  

    roberta_path = "../input/roberta-base/"
MAX_SEQ_LENGTH = 190

TRAIN_BATCH_SIZE = 16

EVAL_BATCH_SIZE = 16

TEST_BATCH_SIZE = 16

LEARNING_RATE = 1e-5

NUM_TRAIN_EPOCHS = 3

BERT_TYPE = "roberta-base-uncased"

max_grad_norm = 1.0

vocab_file = config.roberta_path + "vocab.json"

merge_file = config.roberta_path + "merges.txt"
#tokenizer = tokenizers.BertWordPieceTokenizer(vocab_file,lowercase=True)

tokenizer = tokenizers.ByteLevelBPETokenizer(vocab_file=vocab_file,merges_file=merge_file,lowercase=True,add_prefix_space=True)
raw_train_data = raw_train_data.dropna()
print('The number of training data points are: ',raw_train_data.shape[0])
raw_test_data  = pd.read_csv(basePath+'test.csv')

raw_test_data.head(10)
print('The number of testing data points are: ',raw_test_data.shape[0])
sns.countplot(raw_train_data['sentiment'])
sns.countplot(raw_test_data['sentiment'])
ax = (pd.Series(raw_train_data['sentiment']).value_counts(normalize=True, sort=False)*100).plot.bar()

ax.set(ylabel="Percent")

plt.show()
ax = (pd.Series(raw_test_data['sentiment']).value_counts(normalize=True, sort=False)*100).plot.bar()

ax.set(ylabel="Percent")

plt.show()
def plotWordClouds(df_text,sentiment):

    text = " ".join(str(tmptext) for tmptext in df_text)

    text = text.lower()

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=300,

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

    ).generate(text)

  

    # plot the WordCloud image                        

    plt.figure(figsize = (8, 8), facecolor = None) 

    plt.imshow(wordcloud) 

    plt.axis("off") 

    plt.tight_layout(pad = 0) 

    plt.title('WordCloud - ' + sentiment)

    plt.show()         
subtext = raw_train_data[raw_train_data['sentiment']=='positive']['selected_text']

stopwords = set(STOPWORDS) 

plotWordClouds(subtext,'positive')
subtext = raw_train_data[raw_train_data['sentiment']=='neutral']['selected_text']

plotWordClouds(subtext,'neutral')
subtext = raw_train_data[raw_train_data['sentiment']=='negative']['selected_text']

plotWordClouds(subtext,'negative')
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
def loss_func(out, s_target, e_target):

    criterion = nn.CrossEntropyLoss()

    s_loss = criterion(out[0], s_target)

    e_loss = criterion(out[1], e_target)

    total_loss = s_loss+e_loss

    return total_loss
class BertBaseQA(nn.Module):

    def __init__(self, hidden_size, num_labels,conf):

        super().__init__()

        self.hidden_size = hidden_size

        self.num_labels = num_labels

        self.bert = transformers.RobertaModel.from_pretrained(config.roberta_path,config=conf)

        self.drop_out = nn.Dropout(0.1)

        self.qa_outputs = nn.Linear(self.hidden_size, self.num_labels)

    

    def forward(self, ids, mask, token_ids):



        output = self.bert(

                          input_ids = ids, 

                          attention_mask = mask,

                          token_type_ids = token_ids,

                          )

    

        sequence_output = output[0]   #(None, seq_len, hidden_size)

        sequence_output = self.drop_out(sequence_output)

        logits = self.qa_outputs(sequence_output) #(None, seq_len, hidden_size)*(hidden_size, 2)=(None, seq_len, 2)

        start_logits, end_logits = logits.split(1, dim=-1)    #(None, seq_len, 1), (None, seq_len, 1)

        start_logits = start_logits.squeeze(-1)  #(None, seq_len)

        end_logits = end_logits.squeeze(-1)    #(None, seq_len)





        outputs = (start_logits, end_logits,) 

    

        return outputs
class BertDatasetModule(Dataset):

    def __init__(self, tokenizer, context, question, max_length, text):

        self.context = context

        self.question = question

        self.text = text

        self.tokenizer = tokenizer

        self.max_length = max_length

        

    def __len__(self):

        'Denotes the total number of samples'

        return len(self.context)        

    

    def __getitem__(self, idx):

        'Generates one sample of data'

        context_ = self.context[idx]

        question_ = self.question[idx]

        text_ = self.text[idx]

        

        context_ = " " + " ".join(str(context_).split())

        text_ = " " + " ".join(str(text_).split())



        tok_context = tokenizer.encode(context_)

        tok_question = tokenizer.encode(question_)

        tok_answer = tokenizer.encode(text_)

        

        context_ids = tok_context.ids

        question_ids = tok_question.ids

        answer_ids = tok_answer.ids

        offsets_orig = tok_context.offsets

        #offsets_orig = tok_context.offsets[1:-1]



#        input_ids = [101] + question_ids[1:-1] + [102] + context_ids[1:-1] + [102]

#        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]

#        mask = [1] * len(token_type_ids)        

#        offsets = [(0,0)]*3 + offsets_orig + [(0,0)]

        

        input_ids = [0] + question_ids + [2] + [2] + context_ids + [2]

        token_type_ids = [0, 0, 0, 0] + [0] * (len(context_ids)) + [0]

        mask = [1] * len(token_type_ids)        

        offsets = [(0,0)]*4 + offsets_orig + [(0,0)]

                

        s_pos, e_pos = 0, 0

        for i in range(len(input_ids)):

            if (input_ids[i: i+len(answer_ids)] == answer_ids):

                s_pos = i

                e_pos = i + len(answer_ids) - 1

                break



        assert((s_pos<len(input_ids)) & (e_pos<len(input_ids)) & (s_pos<=e_pos))

        pad_length = self.max_length - len(input_ids)

        if (len(input_ids)<self.max_length):            

            ids = input_ids +([1]*pad_length)

        elif (len(input_ids)>self.max_length):

            ids = input_ids[:self.max_length]

            

        if (len(token_type_ids)<self.max_length):

            token_ids = token_type_ids +([0]*pad_length)

        elif (len(token_type_ids)>self.max_length):

            token_ids = token_type_ids[:self.max_length]   

            

        if (len(mask)<self.max_length):

            mask_ids = mask +([0]*pad_length)

        elif (len(token_type_ids)>self.max_length):

            mask_ids = mask[:self.max_length]

            

        if (len(input_ids)<self.max_length):

            offsets = offsets + ([(0, 0)] * pad_length)

        elif (len(input_ids)>self.max_length):

            offsets = offsets[:self.max_length]            

        

        ids = torch.tensor(ids, dtype = torch.long)

        tt_ids = torch.tensor(token_ids, dtype = torch.long)

        mask_ids = torch.tensor(mask_ids, dtype = torch.long)

        offsets = torch.tensor(offsets, dtype = torch.long)

        start_pos = torch.tensor(s_pos, dtype = torch.long)

        end_pos = torch.tensor(e_pos, dtype = torch.long)

        return {'ids': ids,

            'token_type_ids': tt_ids,

            'mask':mask_ids,

            'start_pos': start_pos,

            'end_pos': end_pos,

            'offsets': offsets,

            'context':context_,

            'question':question_,

            'text':text_}
def train_loop(dataloader, model, optimizer, device, max_grad_norm, scheduler=None):

    model.train()

    for bi, d in enumerate(tqdm_notebook(dataloader, desc="Iteration")):

        ids = d['ids']

        mask_ids = d['mask']

        token_ids = d['token_type_ids']

        start_pos = d['start_pos']

        end_pos = d['end_pos']

        offsets = d['offsets']



        ids = ids.to(device, dtype = torch.long)

        mask_ids = mask_ids.to(device, dtype = torch.long)

        token_ids = token_ids.to(device, dtype = torch.long)

        start_pos = start_pos.to(device, dtype = torch.long)

        end_pos = end_pos.to(device, dtype = torch.long)



        optimizer.zero_grad()

        start_and_end_scores = model(ids, mask_ids, token_ids)

        # start_scores, end_scores = model(ids, token_ids)

        loss = loss_func(start_and_end_scores, start_pos, end_pos)

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        loss.backward()

        optimizer.step()

        if scheduler is not None:

          scheduler.step()

        if bi%100==0:

          print (f"bi: {bi}, loss: {loss}")
def eval_loop(dataloader, model, device):

    model.eval()

    pred_s = None

    pred_e = None

    eval_loss = 0.0

    eval_steps = 0



    for bi, d in enumerate(dataloader):

        ids = d['ids']

        mask_ids = d['mask']

        token_ids = d['token_type_ids']

        start_pos = d['start_pos']

        end_pos = d['end_pos']

        context = d['context']

        question = d['question']

        selected_text = d['text']

        offsets = d['offsets']



        ids = ids.to(device, dtype = torch.long)

        mask_ids = mask_ids.to(device, dtype = torch.long)

        token_ids = token_ids.to(device, dtype = torch.long)

        start_pos = start_pos.to(device, dtype = torch.long)

        end_pos = end_pos.to(device, dtype = torch.long)



        with torch.no_grad():

            start_and_end_scores = model(ids,mask_ids, token_ids)

            loss = loss_func(start_and_end_scores, start_pos, end_pos)

            eval_loss += loss.mean().item()

        

        eval_steps += 1

        

        pred_s = torch.softmax(torch.tensor(start_and_end_scores[0]),dim=1).detach().cpu().numpy()

        pred_e = torch.softmax(torch.tensor(start_and_end_scores[1]),dim=1).detach().cpu().numpy()



    eval_loss = eval_loss/eval_steps

    pred_start = np.argmax(pred_s, axis=1)

    pred_end = np.argmax(pred_e, axis=1)

    

    jaccards=[]        

    for i,tweet in enumerate(context):

        idx_start = pred_start[i]

        idx_end = pred_end[i]   

        if idx_end < idx_start:

            idx_end = idx_start

        filtered_output  = ""        

        for ix in range(idx_start, idx_end + 1):

            filtered_output += tweet[offsets[i][ix][0]: offsets[i][ix][1]]

            if (ix+1) < len(offsets[i]) and offsets[i][ix][1] < offsets[i][ix+1][0]:

                filtered_output += " "

        

        if question[i] == "neutral" or len(tweet.split()) < 2:

            filtered_output = tweet        

                    

        jaccards.append(jaccard(filtered_output,selected_text[i]))  

        

    return eval_loss, pred_start, pred_end, jaccards
test_start = []

test_end = []

mean_jaccard = []
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split

kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

for fold, (tr_ind, val_ind) in enumerate(kf.split(raw_train_data, raw_train_data['sentiment'])):

    train_data = raw_train_data.iloc[tr_ind].reset_index(drop=True)

    val_data = raw_train_data.iloc[val_ind].reset_index(drop=True) 

    

    

    train_dataset = BertDatasetModule(

        tokenizer = tokenizer,

        context = train_data['text'],

        question = train_data['sentiment'],

        max_length = MAX_SEQ_LENGTH,

        text = train_data['selected_text']

    )



    train_dataloader = DataLoader(train_dataset, batch_size = TRAIN_BATCH_SIZE, shuffle=True)    

    

    eval_dataset = BertDatasetModule(

        tokenizer = tokenizer,

        context = val_data['text'],

        question = val_data['sentiment'],

        max_length = MAX_SEQ_LENGTH,

        text = val_data['selected_text']

    ) 



    eval_dataloader = DataLoader(eval_dataset, batch_size = EVAL_BATCH_SIZE, shuffle=False)    

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)



    model_config = transformers.RobertaConfig.from_pretrained(config.roberta_path)

    model = BertBaseQA(768, 2,model_config).to(device)



    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)



    NUM_TRAIN_STEPS = int(len(train_dataset)/TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS) 

    scheduler = transformers.get_constant_schedule_with_warmup(

                    optimizer, 

                    num_warmup_steps=500,

                    # num_training_steps=NUM_TRAIN_STEPS,

                    last_epoch=-1)

    for epoch in trange(NUM_TRAIN_EPOCHS):

        train_loop(train_dataloader, model, optimizer, device, max_grad_norm, scheduler)

        

    res = eval_loop(eval_dataloader, model, device)

    val_start = res[1]

    val_end = res[2]

    jaccards = res[3]

              

    mean_jaccard.append(np.mean(jaccards))

    torch.save(model.state_dict(),'model_' + str(fold) + '.pth')
mean_jaccard
predicted_output = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)

model_config = transformers.RobertaConfig.from_pretrained(config.roberta_path)

model_0 = BertBaseQA(768, 2,model_config).to(device)

model_1 = BertBaseQA(768, 2,model_config).to(device)

model_2 = BertBaseQA(768, 2,model_config).to(device)

model_3 = BertBaseQA(768, 2,model_config).to(device)

model_4 = BertBaseQA(768, 2,model_config).to(device)

model_0.load_state_dict(torch.load('model_0.pth'))

model_1.load_state_dict(torch.load('model_1.pth'))

model_2.load_state_dict(torch.load('model_2.pth'))

model_3.load_state_dict(torch.load('model_3.pth'))

model_4.load_state_dict(torch.load('model_4.pth'))

model_0.eval()

model_1.eval()

model_2.eval()

model_3.eval()

model_4.eval()

    

test_dataset = BertDatasetModule(

        tokenizer = tokenizer,

        context = raw_test_data['text'],

        question = raw_test_data['sentiment'],

        max_length = MAX_SEQ_LENGTH,

        text = raw_test_data['text']

    ) 

test_dataloader = DataLoader(test_dataset, batch_size = TEST_BATCH_SIZE, shuffle=False)       



for bi, d in enumerate(test_dataloader):

    ids = d['ids']

    mask_ids = d['mask']

    token_ids = d['token_type_ids']

    start_pos = d['start_pos']

    end_pos = d['end_pos']

    context = d['context']

    question = d['question']

    selected_text = d['text']

    offsets = d['offsets']



    ids = ids.to(device, dtype = torch.long)

    mask_ids = mask_ids.to(device, dtype = torch.long)

    token_ids = token_ids.to(device, dtype = torch.long)

    start_pos = start_pos.to(device, dtype = torch.long)

    end_pos = end_pos.to(device, dtype = torch.long)

    

    with torch.no_grad():

        start_and_end_scores0 = model_0(ids,mask_ids, token_ids)        

        start_and_end_scores1 = model_1(ids,mask_ids, token_ids)        

        start_and_end_scores2 = model_2(ids,mask_ids, token_ids)   

        start_and_end_scores3 = model_2(ids,mask_ids, token_ids)   

        start_and_end_scores4 = model_3(ids,mask_ids, token_ids)   

        

        start_scores = (start_and_end_scores0[0]+start_and_end_scores1[0]+start_and_end_scores2[0]+start_and_end_scores3[0]+start_and_end_scores4[0])/5

        end_scores = (start_and_end_scores0[1]+start_and_end_scores1[1]+start_and_end_scores2[1]+start_and_end_scores3[1]+start_and_end_scores4[1])/5

        

    pred_s = torch.softmax(torch.tensor(start_scores),dim=1).detach().cpu().numpy()

    pred_e = torch.softmax(torch.tensor(end_scores),dim=1).detach().cpu().numpy()

        

    pred_start = np.argmax(pred_s, axis=1)

    pred_end = np.argmax(pred_e, axis=1)      

           

    for i,tweet in enumerate(context):

        idx_start = pred_start[i]

        idx_end = pred_end[i]       

        if idx_end < idx_start:

            idx_end = idx_start

        filtered_output  = ""        

        for ix in range(idx_start, idx_end + 1):

            filtered_output += tweet[offsets[i][ix][0]: offsets[i][ix][1]]

            if (ix+1) < len(offsets[i]) and offsets[i][ix][1] < offsets[i][ix+1][0]:

                filtered_output += " "

        

        if question == "neutral" or len(tweet.split()) < 2:

            filtered_output = tweet

        predicted_output.append(filtered_output)
sample = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")

sample.loc[:, 'selected_text'] = predicted_output

sample.to_csv("submission.csv", index=False)

sample.head(30)