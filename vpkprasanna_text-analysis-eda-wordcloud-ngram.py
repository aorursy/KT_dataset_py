# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_colwidth', -1)

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from matplotlib import rcParams

import seaborn as sns

import nltk

from nltk.corpus import stopwords

stop = stopwords.words('english')

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from collections import Counter

import plotly.express as px

import plotly.figure_factory as ff

import re

import string
train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")

test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")

submission = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")
train.head()
train.isna().sum()
train = train.dropna()
train.sentiment.value_counts()
rcParams["figure.figsize"] = 15,13

train.sentiment.value_counts().plot(kind="pie")
rcParams["figure.figsize"] = 15,10

sns.countplot(x=train["sentiment"],data=train)
def generate_word_cloud(text):

    wordcloud = WordCloud(

        width = 3000,

        height = 2000,

        background_color = 'black').generate(str(text))

    fig = plt.figure(

        figsize = (40, 30),

        facecolor = 'k',

        edgecolor = 'k')

    plt.imshow(wordcloud, interpolation = 'bilinear')

    plt.axis('off')

    plt.tight_layout(pad=0)

    plt.show()
def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = str(text).lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text
train['text'] = train['text'].apply(lambda x:clean_text(x))

train['selected_text'] = train['selected_text'].apply(lambda x:clean_text(x))
train.head()
train['temp_list'] = train['selected_text'].apply(lambda x:str(x).split())

top = Counter([item for sublist in train['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(20))

temp.columns = ['Common_words','count']

temp.style.background_gradient(cmap='Blues')
fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Selected Text', orientation='h', width=700, height=700,color='Common_words')

fig.show()
fig = px.treemap(temp, path=['Common_words'], values='count',title='Tree of Most Common Words')

fig.show()
def remove_stopword(x):

    return [w for w in x if not w in stop]
train['temp_list'] = train['temp_list'].apply(lambda x:remove_stopword(x))
top = Counter([item for sublist in train['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(20))

temp.columns = ['Common_words','count']

temp.style.background_gradient(cmap='Purples')
fig = px.treemap(temp, path=['Common_words'], values='count',title='Tree of Most Common Words')

fig.show()
train_bigram = (pd.Series(nltk.ngrams(train["selected_text"], 2)).value_counts())[:20]
train_bigram.sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))

plt.title('20 Most Frequently Occuring Bigrams')

plt.ylabel('Bigram')

plt.xlabel('# of Occurances')

train_text = train.selected_text[:100].values

generate_word_cloud(train_text)
# train_trigram = pd.Series(nltk.ngrams(train["text"], 3)).value_counts()[:20]

# train_trigram.sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))

# plt.title('20 Most Frequently Occuring Bigrams')

# plt.ylabel('Trigram')

# plt.xlabel('# of Occurances')
positive_train = train[train["sentiment"]=="positive"]

negative_train = train[train["sentiment"]=="negative"]

neutral_train = train[train["sentiment"]=="neutral"]
def tokenizeandstopwords(text):

    tokens = nltk.word_tokenize(text)

    # taken only words (not punctuation)

    token_words = [w for w in tokens if w.isalpha()]

    meaningful_words = [w for w in token_words if not w in stop]

    joined_words = ( " ".join(meaningful_words))

    return joined_words
positive_train["selected_text"] = positive_train["selected_text"].apply(clean_text)

negative_train["selected_text"] = negative_train["selected_text"].apply(clean_text)

neutral_train["selected_text"] = neutral_train["selected_text"].apply(clean_text)
positive_train["selected_text"] = positive_train["selected_text"].apply(tokenizeandstopwords)

negative_train["selected_text"] = negative_train["selected_text"].apply(tokenizeandstopwords)

neutral_train["selected_text"] = neutral_train["selected_text"].apply(tokenizeandstopwords)
positive_bigram = (pd.Series(nltk.ngrams(positive_train["selected_text"], 2)).value_counts())[:25]
positive_bigram.sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))

plt.title('25 Most Frequently Occuring Bigrams')

plt.ylabel('Bigram')

plt.xlabel('# of Occurances')

positive_train['temp_list'] = positive_train['selected_text'].apply(lambda x:str(x).split())

positive_train['temp_list'] = positive_train['temp_list'].apply(lambda x:remove_stopword(x))

positive_top = Counter([item for sublist in positive_train['temp_list'] for item in sublist])

positive_temp = pd.DataFrame(positive_top.most_common(20))

positive_temp.columns = ['Common_words','count']

positive_temp.style.background_gradient(cmap='Blues')
fig = px.treemap(positive_temp, path=['Common_words'], values='count',title='Tree of Most Common Positive Words')

fig.show()
positive_text = positive_train.selected_text[:100].values

generate_word_cloud(positive_text)
negative_train['temp_list'] = negative_train['selected_text'].apply(lambda x:str(x).split())

negative_train['temp_list'] = negative_train['temp_list'].apply(lambda x:remove_stopword(x))

negative_top = Counter([item for sublist in negative_train['temp_list'] for item in sublist])

negative_temp = pd.DataFrame(negative_top.most_common(20))

negative_temp.columns = ['Common_words','count']

negative_temp.style.background_gradient(cmap='Blues')
fig = px.treemap(negative_temp, path=['Common_words'], values='count',title='Tree of Most Common  Negative Words')

fig.show()
negative_bigram = (pd.Series(nltk.ngrams(negative_train["selected_text"], 2)).value_counts())[:25]

negative_bigram.sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))

plt.title('25 Most Frequently Occuring Bigrams')

plt.ylabel('Bigram')

plt.xlabel('# of Occurances')
negative_text = negative_train.selected_text[:100].values

generate_word_cloud(negative_text)
neutral_train['temp_list'] = neutral_train['selected_text'].apply(lambda x:str(x).split())

neutral_train['temp_list'] = neutral_train['temp_list'].apply(lambda x:remove_stopword(x))

neutral_top = Counter([item for sublist in neutral_train['temp_list'] for item in sublist])

neutral_temp = pd.DataFrame(neutral_top.most_common(20))

neutral_temp.columns = ['Common_words','count']

neutral_temp.style.background_gradient(cmap='Blues')
fig = px.treemap(neutral_temp, path=['Common_words'], values='count',title='Tree of Most Common Neutral Words')

fig.show()
neutral_bigram = (pd.Series(nltk.ngrams(neutral_train["selected_text"], 2)).value_counts())[:25]

neutral_bigram.sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))

plt.title('25 Most Frequently Occuring Bigrams')

plt.ylabel('Bigram')

plt.xlabel('# of Occurances')

neutral_text = neutral_train.selected_text[:100].values

generate_word_cloud(neutral_text)
import numpy as np             # for algebric functions

import pandas as pd            # to handle dataframes

import os                      # to import files 

#!pip install transformers

import transformers            # Transformers (pytorch-transformers /pytorch-pretrained-bert) provides general-purpose architectures (BERT, RoBERTa,..)

import tokenizers              # A tokenizer is in charge of preparing the inputs for a model. 

import string                  

import torch                   # pytorch

import torch.nn as nn   

from torch.nn import functional as F

from tqdm import tqdm          # TQDM is a progress bar library

import re                      # regular expression

import json

import requests
MAX_LEN = 192

TRAIN_BATCH_SIZE = 32

VALID_BATCH_SIZE = 8

EPOCHS = 5

ROBERTA_PATH = 'roberta-base'




# pre_voc_file = transformers.RobertaTokenizer.pretrained_vocab_files_map

# merges_file  = pre_voc_file.get('merges_file').get(ROBERTA_PATH)

# vocab_file = pre_voc_file.get('vocab_file').get(ROBERTA_PATH)

# model_bin = transformers.modeling_roberta.ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP.get(ROBERTA_PATH)
# json_f = requests.get(vocab_file) 

# txt_f = requests.get(merges_file) 

# mod_bin = requests.get(model_bin)



# data = json_f.json()
# with open('vocab.json', 'w') as f: json.dump(data, f)
# open('merge.txt', 'wb').write(txt_f.content) 

# open('model.bin', 'wb').write(mod_bin.content)
# TOKENIZER = tokenizers.ByteLevelBPETokenizer(vocab_file=f"../input/roberta-vocab-file/vocab.json", 

#                                              merges_file=f"../input/roberta-vocab-file/merge.txt", 

#                                              lowercase=True,

#                                              add_prefix_space=True)
# class TweetModel(transformers.BertPreTrainedModel):

#     def __init__(self, conf):

#         super(TweetModel, self).__init__(conf)

#         self.roberta = transformers.RobertaModel.from_pretrained("roberta-base", config=conf)

#         self.drop_out = nn.Dropout(0.1)

#         self.l0 = nn.Linear(768 * 2, 2)

#         torch.nn.init.normal_(self.l0.weight, std=0.02)

    

#     def forward(self, ids, mask, token_type_ids):

#         _, _, out = self.roberta(

#             ids,

#             attention_mask=mask,

#             token_type_ids=token_type_ids

#         )



#         out = torch.cat((out[-1], out[-2]), dim=-1)

#         out = self.drop_out(out)

#         logits = self.l0(out)



#         start_logits, end_logits = logits.split(1, dim=-1)



#         start_logits = start_logits.squeeze(-1)

#         end_logits = end_logits.squeeze(-1)



#         return start_logits, end_logits

# def process_data(tweet, selected_text, sentiment, tokenizer, max_len):

#     tweet = " " + " ".join(str(tweet).split())

#     selected_text = " " + " ".join(str(selected_text).split())



#     len_st = len(selected_text) - 1

#     idx0 = None

#     idx1 = None



#     for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):

#         if " " + tweet[ind: ind+len_st] == selected_text:

#             idx0 = ind

#             idx1 = ind + len_st - 1

#             break



#     char_targets = [0] * len(tweet)

#     if idx0 != None and idx1 != None:

#         for ct in range(idx0, idx1 + 1):

#             char_targets[ct] = 1

    

#     tok_tweet = tokenizer.encode(tweet)

#     input_ids_orig = tok_tweet.ids

#     tweet_offsets = tok_tweet.offsets

    

#     target_idx = []

#     for j, (offset1, offset2) in enumerate(tweet_offsets):

#         if sum(char_targets[offset1: offset2]) > 0:

#             target_idx.append(j)

    

#     targets_start = target_idx[0]

#     targets_end = target_idx[-1]



#     sentiment_id = {

#         'positive': 1313,

#         'negative': 2430,

#         'neutral': 7974

#     }

    

#     input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]

#     token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)

#     mask = [1] * len(token_type_ids)

#     tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]

#     targets_start += 4

#     targets_end += 4



#     padding_length = max_len - len(input_ids)

#     if padding_length > 0:

#         input_ids = input_ids + ([1] * padding_length)

#         mask = mask + ([0] * padding_length)

#         token_type_ids = token_type_ids + ([0] * padding_length)

#         tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)

    

#     return {

#         'ids': input_ids,

#         'mask': mask,

#         'token_type_ids': token_type_ids,

#         'targets_start': targets_start,

#         'targets_end': targets_end,

#         'orig_tweet': tweet,

#         'orig_selected': selected_text,

#         'sentiment': sentiment,

#         'offsets': tweet_offsets

#     }
# class TweetDataset:

#     def __init__(self, tweet, sentiment, selected_text):

#         self.tweet = tweet

#         self.sentiment = sentiment

#         self.selected_text = selected_text

#         self.tokenizer = TOKENIZER

#         self.max_len = MAX_LEN

    

#     def __len__(self):

#         return len(self.tweet)



#     def __getitem__(self, item):

#         data = process_data(

#             self.tweet[item], 

#             self.selected_text[item], 

#             self.sentiment[item],

#             self.tokenizer,

#             self.max_len

#         )



#         return {

#             'ids': torch.tensor(data["ids"], dtype=torch.long),

#             'mask': torch.tensor(data["mask"], dtype=torch.long),

#             'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),

#             'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),

#             'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),

#             'orig_tweet': data["orig_tweet"],

#             'orig_selected': data["orig_selected"],

#             'sentiment': data["sentiment"],

#             'offsets': torch.tensor(data["offsets"], dtype=torch.long)

#         }

# def calculate_jaccard_score(

#     original_tweet, 

#     target_string, 

#     sentiment_val, 

#     idx_start, 

#     idx_end, 

#     offsets,

#     verbose=False):

    

#     if idx_end < idx_start:

#         idx_end = idx_start

    

#     filtered_output  = ""

#     for ix in range(idx_start, idx_end + 1):

#         filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]

#         if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:

#             filtered_output += " "



#     if sentiment_val == "neutral" or len(original_tweet.split()) < 2:

#         filtered_output = original_tweet



#     if sentiment_val != "neutral" and verbose == True:

#         if filtered_output.strip().lower() != target_string.strip().lower():

#             print("********************************")

#             print(f"Output= {filtered_output.strip()}")

#             print(f"Target= {target_string.strip()}")

#             print(f"Tweet= {original_tweet.strip()}")

#             print("********************************")



#     jac = 0

#     return jac, filtered_output
# df_test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")

# df_test.loc[:, "selected_text"] = df_test.text.values

# device = torch.device("cuda")

# model_config = transformers.RobertaConfig.from_pretrained('../input/roberta-vocab-file/config.json')  # to download from internet

# model_config.output_hidden_states = True
# TweetDataset(tweet=df_test.text.values,

#              sentiment=df_test.sentiment.values,

#              selected_text=df_test.selected_text.values)
# model = TweetModel(conf=model_config)

# model.to(device)

# model.eval()

# final_output = []

# test_dataset = TweetDataset(

#         tweet=df_test.text.values,

#         sentiment=df_test.sentiment.values,

#         selected_text=df_test.selected_text.values

#     )



# data_loader = torch.utils.data.DataLoader(

#     test_dataset,

#     shuffle=False,

#     batch_size=VALID_BATCH_SIZE,

#     num_workers=0

# )

# with torch.no_grad():

#     tk0 = tqdm(data_loader, total=len(data_loader))

#     for bi, d in enumerate(tk0):

#         ids = d["ids"]

#         token_type_ids = d["token_type_ids"]

#         mask = d["mask"]

#         sentiment = d["sentiment"]

#         orig_selected = d["orig_selected"]

#         orig_tweet = d["orig_tweet"]

#         targets_start = d["targets_start"]

#         targets_end = d["targets_end"]

#         offsets = d["offsets"].numpy()



#         ids            = ids.to(device, dtype=torch.long)

#         token_type_ids = token_type_ids.to(device, dtype=torch.long)

#         mask           = mask.to(device, dtype=torch.long)

#         targets_start  = targets_start.to(device, dtype=torch.long)

#         targets_end    = targets_end.to(device, dtype=torch.long)



#         outputs_start1, outputs_end1 = model(

#             ids=ids,

#             mask=mask,

#             token_type_ids=token_type_ids

#         )



#         outputs_start = outputs_start1

#         outputs_end = outputs_end1

        

#         outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()

#         outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()

#         jaccard_scores = []

#         for px, tweet in enumerate(orig_tweet):

#           selected_tweet = orig_selected[px]

#           tweet_sentiment = sentiment[px]

#           _, output_sentence = calculate_jaccard_score(original_tweet=tweet,

#                                                        target_string=selected_tweet,

#                                                        sentiment_val=tweet_sentiment,

#                                                        idx_start=np.argmax(outputs_start[px, :]),

#                                                        idx_end=np.argmax(outputs_end[px, :]),

#                                                        offsets=offsets[px])

#           final_output.append(output_sentence)
# sample = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")

# sample.loc[:, 'selected_text'] = final_output

# sample.to_csv("submission.csv", index=False)

class AverageMeter:

    """

    Computes and stores the average and current value

    """

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





def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
import os

import torch

import pandas as pd

import torch.nn as nn

import numpy as np

import torch.nn.functional as F

from torch.optim import lr_scheduler



from sklearn import model_selection

from sklearn import metrics

import transformers

import tokenizers

from transformers import AdamW

from transformers import get_linear_schedule_with_warmup

from tqdm.autonotebook import tqdm

class config:

    MAX_LEN = 128

    TRAIN_BATCH_SIZE = 64

    VALID_BATCH_SIZE = 16

    EPOCHS = 5

    BERT_PATH = "../input/bert-base-uncased/"

    MODEL_PATH = "model.bin"

    TRAINING_FILE = "../input/tweet-train-folds/train_folds.csv"

    TOKENIZER = tokenizers.BertWordPieceTokenizer(

        f"{BERT_PATH}/vocab.txt", 

        lowercase=True

    )




def process_data(tweet, selected_text, sentiment, tokenizer, max_len):

    len_st = len(selected_text)

    idx0 = None

    idx1 = None

    for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):

        if tweet[ind: ind+len_st] == selected_text:

            idx0 = ind

            idx1 = ind + len_st - 1

            break



    char_targets = [0] * len(tweet)

    if idx0 != None and idx1 != None:

        for ct in range(idx0, idx1 + 1):

            char_targets[ct] = 1

    

    tok_tweet = tokenizer.encode(tweet)

    input_ids_orig = tok_tweet.ids[1:-1]

    tweet_offsets = tok_tweet.offsets[1:-1]

    

    target_idx = []

    for j, (offset1, offset2) in enumerate(tweet_offsets):

        if sum(char_targets[offset1: offset2]) > 0:

            target_idx.append(j)

    

    targets_start = target_idx[0]

    targets_end = target_idx[-1]



    sentiment_id = {

        'positive': 3893,

        'negative': 4997,

        'neutral': 8699

    }

    

    input_ids = [101] + [sentiment_id[sentiment]] + [102] + input_ids_orig + [102]

    token_type_ids = [0, 0, 0] + [1] * (len(input_ids_orig) + 1)

    mask = [1] * len(token_type_ids)

    tweet_offsets = [(0, 0)] * 3 + tweet_offsets + [(0, 0)]

    targets_start += 3

    targets_end += 3



    padding_length = max_len - len(input_ids)

    if padding_length > 0:

        input_ids = input_ids + ([0] * padding_length)

        mask = mask + ([0] * padding_length)

        token_type_ids = token_type_ids + ([0] * padding_length)

        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)

    

    return {

        'ids': input_ids,

        'mask': mask,

        'token_type_ids': token_type_ids,

        'targets_start': targets_start,

        'targets_end': targets_end,

        'orig_tweet': tweet,

        'orig_selected': selected_text,

        'sentiment': sentiment,

        'offsets': tweet_offsets

    }



class TweetDataset:

    def __init__(self, tweet, sentiment, selected_text):

        self.tweet = tweet

        self.sentiment = sentiment

        self.selected_text = selected_text

        self.tokenizer = config.TOKENIZER

        self.max_len = config.MAX_LEN

    

    def __len__(self):

        return len(self.tweet)



    def __getitem__(self, item):

        data = process_data(

            self.tweet[item], 

            self.selected_text[item], 

            self.sentiment[item],

            self.tokenizer,

            self.max_len

        )



        return {

            'ids': torch.tensor(data["ids"], dtype=torch.long),

            'mask': torch.tensor(data["mask"], dtype=torch.long),

            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),

            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),

            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),

            'orig_tweet': data["orig_tweet"],

            'orig_selected': data["orig_selected"],

            'sentiment': data["sentiment"],

            'offsets': torch.tensor(data["offsets"], dtype=torch.long)

        }




class TweetModel(transformers.BertPreTrainedModel):

    def __init__(self, conf):

        super(TweetModel, self).__init__(conf)

        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH, config=conf)

        self.drop_out = nn.Dropout(0.1)

        self.l0 = nn.Linear(768 * 2, 2)

        torch.nn.init.normal_(self.l0.weight, std=0.02)

    

    def forward(self, ids, mask, token_type_ids):

        _, _, out = self.bert(

            ids,

            attention_mask=mask,

            token_type_ids=token_type_ids

        )



        out = torch.cat((out[-1], out[-2]), dim=-1)

        out = self.drop_out(out)

        logits = self.l0(out)



        start_logits, end_logits = logits.split(1, dim=-1)



        start_logits = start_logits.squeeze(-1)

        end_logits = end_logits.squeeze(-1)



        return start_logits, end_logits







def loss_fn(start_logits, end_logits, start_positions, end_positions):

    loss_fct = nn.CrossEntropyLoss()

    start_loss = loss_fct(start_logits, start_positions)

    end_loss = loss_fct(end_logits, end_positions)

    total_loss = (start_loss + end_loss)

    return total_loss



def train_fn(data_loader, model, optimizer, device, scheduler=None):

    model.train()

    losses = AverageMeter()

    jaccards = AverageMeter()



    tk0 = tqdm(data_loader, total=len(data_loader))

    

    for bi, d in enumerate(tk0):



        ids = d["ids"]

        token_type_ids = d["token_type_ids"]

        mask = d["mask"]

        targets_start = d["targets_start"]

        targets_end = d["targets_end"]

        sentiment = d["sentiment"]

        orig_selected = d["orig_selected"]

        orig_tweet = d["orig_tweet"]

        targets_start = d["targets_start"]

        targets_end = d["targets_end"]

        offsets = d["offsets"]



        ids = ids.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        targets_start = targets_start.to(device, dtype=torch.long)

        targets_end = targets_end.to(device, dtype=torch.long)



        model.zero_grad()

        outputs_start, outputs_end = model(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids,

        )

        loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)

        loss.backward()

        optimizer.step()

        scheduler.step()



        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()

        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()

        jaccard_scores = []

        for px, tweet in enumerate(orig_tweet):

            selected_tweet = orig_selected[px]

            tweet_sentiment = sentiment[px]

            jaccard_score, _ = calculate_jaccard_score(

                original_tweet=tweet,

                target_string=selected_tweet,

                sentiment_val=tweet_sentiment,

                idx_start=np.argmax(outputs_start[px, :]),

                idx_end=np.argmax(outputs_end[px, :]),

                offsets=offsets[px]

            )

            jaccard_scores.append(jaccard_score)



        jaccards.update(np.mean(jaccard_scores), ids.size(0))

        losses.update(loss.item(), ids.size(0))

        tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)
def calculate_jaccard_score(

    original_tweet, 

    target_string, 

    sentiment_val, 

    idx_start, 

    idx_end, 

    offsets,

    verbose=False):

    

    if idx_end < idx_start:

        idx_end = idx_start

    

    filtered_output  = ""

    for ix in range(idx_start, idx_end + 1):

        filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]

        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:

            filtered_output += " "



    if sentiment_val == "neutral" or len(original_tweet.split()) < 2:

        filtered_output = original_tweet



    jac = jaccard(target_string.strip(), filtered_output.strip())

    return jac, filtered_output





def eval_fn(data_loader, model, device):

    model.eval()

    losses = AverageMeter()

    jaccards = AverageMeter()

    

    with torch.no_grad():

        tk0 = tqdm(data_loader, total=len(data_loader))

        for bi, d in enumerate(tk0):

            ids = d["ids"]

            token_type_ids = d["token_type_ids"]

            mask = d["mask"]

            sentiment = d["sentiment"]

            orig_selected = d["orig_selected"]

            orig_tweet = d["orig_tweet"]

            targets_start = d["targets_start"]

            targets_end = d["targets_end"]

            offsets = d["offsets"].numpy()



            ids = ids.to(device, dtype=torch.long)

            token_type_ids = token_type_ids.to(device, dtype=torch.long)

            mask = mask.to(device, dtype=torch.long)

            targets_start = targets_start.to(device, dtype=torch.long)

            targets_end = targets_end.to(device, dtype=torch.long)



            outputs_start, outputs_end = model(

                ids=ids,

                mask=mask,

                token_type_ids=token_type_ids

            )

            loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)

            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()

            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()

            jaccard_scores = []

            for px, tweet in enumerate(orig_tweet):

                selected_tweet = orig_selected[px]

                tweet_sentiment = sentiment[px]

                jaccard_score, _ = calculate_jaccard_score(

                    original_tweet=tweet,

                    target_string=selected_tweet,

                    sentiment_val=tweet_sentiment,

                    idx_start=np.argmax(outputs_start[px, :]),

                    idx_end=np.argmax(outputs_end[px, :]),

                    offsets=offsets[px]

                )

                jaccard_scores.append(jaccard_score)



            jaccards.update(np.mean(jaccard_scores), ids.size(0))

            losses.update(loss.item(), ids.size(0))

            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)

    

    print(f"Jaccard = {jaccards.avg}")

    return jaccards.avg
def run(fold):

    dfx = pd.read_csv(config.TRAINING_FILE)



    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)

    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

    

    train_dataset = TweetDataset(

        tweet=df_train.text.values,

        sentiment=df_train.sentiment.values,

        selected_text=df_train.selected_text.values

    )



    train_data_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=config.TRAIN_BATCH_SIZE,

        num_workers=4

    )



    valid_dataset = TweetDataset(

        tweet=df_valid.text.values,

        sentiment=df_valid.sentiment.values,

        selected_text=df_valid.selected_text.values

    )



    valid_data_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=config.VALID_BATCH_SIZE,

        num_workers=2

    )



    device = torch.device("cuda")

    model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)

    model_config.output_hidden_states = True

    model = TweetModel(conf=model_config)

    model.to(device)



    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [

        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},

        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},

    ]

    optimizer = AdamW(optimizer_parameters, lr=3e-5)

    scheduler = get_linear_schedule_with_warmup(

        optimizer, 

        num_warmup_steps=0, 

        num_training_steps=num_train_steps

    )



    es = EarlyStopping(patience=2, mode="max")

    print(f"Training is Starting for fold={fold}")

    

    # I'm training only for 3 epochs even though I specified 5!!!

    for epoch in range(3):

        train_fn(train_data_loader, model, optimizer, device, scheduler=scheduler)

        jaccard = eval_fn(valid_data_loader, model, device)

        print(f"Jaccard Score = {jaccard}")

        es(jaccard, model, model_path=f"model_{fold}.bin")

        if es.early_stop:

            print("Early stopping")

            break

run(fold=0)
run(fold=1)
run(fold=2)
run(fold=3)
run(fold=4)
df_test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")

df_test.loc[:, "selected_text"] = df_test.text.values




device = torch.device("cuda")

model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)

model_config.output_hidden_states = True



model1 = TweetModel(conf=model_config)

model1.to(device)

model1.load_state_dict(torch.load("model_0.bin"))

model1.eval()



model2 = TweetModel(conf=model_config)

model2.to(device)

model2.load_state_dict(torch.load("model_1.bin"))

model2.eval()



model3 = TweetModel(conf=model_config)

model3.to(device)

model3.load_state_dict(torch.load("model_2.bin"))

model3.eval()



model4 = TweetModel(conf=model_config)

model4.to(device)

model4.load_state_dict(torch.load("model_3.bin"))

model4.eval()



model5 = TweetModel(conf=model_config)

model5.to(device)

model5.load_state_dict(torch.load("model_4.bin"))

model5.eval()
final_output = []



test_dataset = TweetDataset(

        tweet=df_test.text.values,

        sentiment=df_test.sentiment.values,

        selected_text=df_test.selected_text.values

)



data_loader = torch.utils.data.DataLoader(

    test_dataset,

    shuffle=False,

    batch_size=config.VALID_BATCH_SIZE,

    num_workers=1

)



with torch.no_grad():

    tk0 = tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk0):

        ids = d["ids"]

        token_type_ids = d["token_type_ids"]

        mask = d["mask"]

        sentiment = d["sentiment"]

        orig_selected = d["orig_selected"]

        orig_tweet = d["orig_tweet"]

        targets_start = d["targets_start"]

        targets_end = d["targets_end"]

        offsets = d["offsets"].numpy()



        ids = ids.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        targets_start = targets_start.to(device, dtype=torch.long)

        targets_end = targets_end.to(device, dtype=torch.long)



        outputs_start1, outputs_end1 = model1(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

        )

        

        outputs_start2, outputs_end2 = model2(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

        )

        

        outputs_start3, outputs_end3 = model3(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

        )

        

        outputs_start4, outputs_end4 = model4(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

        )

        

        outputs_start5, outputs_end5 = model5(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

        )

        outputs_start = (

            outputs_start1 

            + outputs_start2 

            + outputs_start3 

            + outputs_start4 

            + outputs_start5

        ) / 5

        outputs_end = (

            outputs_end1 

            + outputs_end2 

            + outputs_end3 

            + outputs_end4 

            + outputs_end5

        ) / 5

        

        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()

        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()



        for px, tweet in enumerate(orig_tweet):

            selected_tweet = orig_selected[px]

            tweet_sentiment = sentiment[px]

            _, output_sentence = calculate_jaccard_score(

                original_tweet=tweet,

                target_string=selected_tweet,

                sentiment_val=tweet_sentiment,

                idx_start=np.argmax(outputs_start[px, :]),

                idx_end=np.argmax(outputs_end[px, :]),

                offsets=offsets[px]

            )

            final_output.append(output_sentence)

def post_process(selected):

    return " ".join(set(selected.lower().split()))
sample = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")

sample.loc[:, 'selected_text'] = final_output

sample.selected_text = sample.selected_text.map(post_process)

sample.to_csv("submission.csv", index=False)