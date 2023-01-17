import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
TRAIN1_PATH = "/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv"

TRAIN2_PATH = "/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train-processed-seqlen128.csv"

VALID_PATH = "/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv"

TEST_PATH = "/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv"



train1_df = pd.read_csv(TRAIN1_PATH, usecols=['comment_text', 'toxic']).fillna('none')

train2_df = pd.read_csv(TRAIN2_PATH, usecols=['comment_text', 'toxic']).fillna('none')

train_full_df = pd.concat([train1_df, train2_df], axis=0).reset_index(drop=True)

train_df = train_full_df.sample(frac=1).reset_index(drop=True)

valid_df = pd.read_csv(VALID_PATH, usecols=['comment_text', 'toxic'])
train_df = train_df.head(400000)

train_df.comment_text.values.shape
from transformers import XLMRobertaTokenizer



xlm_roberta_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
def preprocess(text):

    text = str(text).strip().lower()

    return " ".join(text.split())





def regular_encode(text, tokenizer, max_length):

    output = tokenizer.batch_encode_plus(

        text,

        return_token_type_ids=False,

        pad_to_max_length=True,

        add_special_tokens=False,

        max_length=max_length)

    

    return np.array(output['input_ids']), np.array(output['attention_mask'])
train_encodings = regular_encode(

    text=train_df.comment_text.values, 

    tokenizer=xlm_roberta_tokenizer,

    max_length=192)



train_input_ids, train_attention_mask = train_encodings



valid_encodings = regular_encode(

    text=valid_df.comment_text.values,

    tokenizer=xlm_roberta_tokenizer,

    max_length=192)



valid_input_ids, valid_attention_mask = valid_encodings





np.save('train_input_ids', train_input_ids)

np.save('train_attention_mask', train_attention_mask)

np.save('train_targets', train_df.toxic.values)



np.save('valid_input_ids', valid_input_ids)

np.save('valid_attention_mask', valid_attention_mask)

np.save('valid_targets', valid_df.toxic.values)



print(f"Complete!")
train_input_ids.shape, train_attention_mask.shape