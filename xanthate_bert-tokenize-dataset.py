import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm
from multiprocessing.pool import Pool

import os
import sys
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from transformers import BertTokenizer
from transformers import XLMRobertaTokenizer
TRAIN1_PATH = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv'
TRAIN2_PATH = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv'
VALID_PATH = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv'
TEST_PATH = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv'

train1_df = pd.read_csv(TRAIN1_PATH, usecols=['comment_text', 'toxic']).fillna('none')
train2_df = pd.read_csv(TRAIN2_PATH, usecols=['comment_text', 'toxic']).fillna('none')
trainfull_df = pd.concat([train1_df, train2_df], axis=0).reset_index(drop=True)

train_df = trainfull_df.sample(frac=1, random_state=42)
valid_df = pd.read_csv(VALID_PATH, usecols=['comment_text', 'toxic'])
test_df = pd.read_csv(TEST_PATH, usecols=['content'])

train1_df.shape, train2_df.shape, valid_df.shape, test_df.shape
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#xlm_roberta_tokenizer = XLMRobertaTokenizer.from_pretrained('')

MAX_LENGTH = 192
#num_cores = 4
train_N = 800000
test_N = test_df.shape[0]
valid_N = valid_df.shape[0]


# get dataframes
train_df = train_df.head(train_N)
test_df = test_df
valid_df = valid_df


# remove wierd spaces and convert to lower case
def preprocessing(text):
    text = str(text).strip().lower()
    return " ".join(text.split())


# encode string for each subprocess
def token_encoding(t, target, tokenizer, max_length):
    # there is no target for the test case
    if target is True:
        texts = t[0]
        targets = t[1]
    else:
        texts = t
    
    input_ids = []
    token_type_ids = []
    attention_mask = []
    
    for i in tqdm(range(0, len(texts))):
        text = preprocessing(texts[i])
        inputs = tokenizer.encode_plus(text,
                                       None,
                                       pad_to_max_length=True, 
                                       max_length=max_length)
        
        input_ids.append(inputs['input_ids'])
        token_type_ids.append(inputs['token_type_ids'])
        attention_mask.append(inputs['attention_mask'])
    
    if target is True:
        return np.array(input_ids), np.array(token_type_ids), np.array(attention_mask), np.array(targets)
    
    else:
        return np.array(input_ids), np.array(token_type_ids), np.array(attention_mask)


def encoding_process(df, N, num_cores, tokenizer, max_length):
    
    text1 = df.comment_text.values[:int(N/4)]
    target1 = df.toxic.values[:int(N/4)]
    
    text2 = df.comment_text.values[int(N/4):int(N/2)]
    target2 = df.toxic.values[int(N/4):int(N/2)]
    
    text3 = df.comment_text.values[int(N/2):int(0.75 * N)]
    target3 = df.toxic.values[int(N/2):int(0.75 * N)]
    
    text4 = df.comment_text.values[int(0.75 * N):]
    target4 = df.toxic.values[int(0.75 * N):]
    
    process_pool = Pool(num_cores)
    
    chunk1 = ((text1, target1), True, tokenizer, max_length)
    chunk2 = ((text2, target2), True, tokenizer, max_length)
    chunk3 = ((text3, target3), True, tokenizer, max_length)
    chunk4 = ((text4, target4), True, tokenizer, max_length)
    
    chunks = [chunk1, chunk2, chunk3, chunk4]
    
    output = process_pool.starmap(token_encoding, chunks)
    
    input_ids = np.concatenate([out[0] for out in output], axis=0)
    token_type_ids = np.concatenate([out[1] for out in output], axis=0)
    attention_mask = np.concatenate([out[2] for out in output], axis=0)
    targets = np.concatenate([out[3] for out in output], axis=0)
    
    assert input_ids.shape[0] == token_type_ids.shape[0] \
            == attention_mask.shape[0] == targets.shape[0] == N
    
    return input_ids, token_type_ids, attention_mask, targets
    

def test_encoding_process(df, N, num_cores, tokenizer, max_length):
    
    text1 = df.content.values[:int(N/4)]
    
    text2 = df.content.values[int(N/4):int(N/2)]
    
    text3 = df.content.values[int(N/2):int(0.75 * N)]
    
    text4 = df.content.values[int(0.75 * N):]
    
    process_pool = Pool(num_cores)
    
    chunk1 = (text1, False, tokenizer, max_length)
    chunk2 = (text2, False, tokenizer, max_length)
    chunk3 = (text3, False, tokenizer, max_length)
    chunk4 = (text4, False, tokenizer, max_length)
    
    chunks = [chunk1, chunk2, chunk3, chunk4]
    
    output = process_pool.starmap(token_encoding, chunks)
    
    input_ids = np.concatenate([out[0] for out in output], axis=0)
    token_type_ids = np.concatenate([out[1] for out in output], axis=0)
    attention_mask = np.concatenate([out[2] for out in output], axis=0)
    
    assert input_ids.shape[0] == token_type_ids.shape[0] == attention_mask.shape[0] == N
    
    return input_ids, token_type_ids, attention_mask


def save_data(compressed=False):
    if compressed is True:
        np.savez_compressed('train-df-compressed-input-ids.npz', train_input_ids)
        np.savez_compressed('train-df-compressed-attention-mask.npz', train_attention_mask)
        np.savez_compressed('train-df-compressed-token-type-ids.npz', train_token_type_ids)
        np.savez_compressed('train-df-compressed-targets.npz', train_targets)
        
        # valid
        np.savez_compressed('valid-df-compressed-input-ids.npz', valid_input_ids)
        np.savez_compressed('valid-df-compressed-attention-mask.npz', valid_attention_mask)
        np.savez_compressed('valid-df-compressed-token-type-ids.npz', valid_token_type_ids)
        np.savez_compressed('valid-df-compressed-targets.npz', valid_targets)
        
        # test
        np.savez_compressed('test-df-compressed-input-ids.npz', test_input_ids)
        np.savez_compressed('test-df-compressed-attention-mask.npz', test_attention_mask)
        np.savez_compressed('test-df-compressed-token-type-ids.npz', test_token_type_ids)

    else:
        np.save('train-df-input-ids.npy', input_ids)
        np.save('train-df-attention-mask.npy', attention_mask)
        np.save('train-df-token-type-ids', token_type_ids)
        np.save('train-df-targets', targets)
        
        
train_input_ids, train_token_type_ids, train_attention_mask, train_targets \
                        = encoding_process(
                                    df=train_df,
                                    N=train_N,
                                    num_cores=4,
                                    tokenizer=bert_tokenizer,
                                    max_length=MAX_LENGTH)

print("Training data encoding complete.")


valid_input_ids, valid_token_type_ids, valid_attention_mask, valid_targets \
                        = encoding_process(
                                    df=valid_df,
                                    N=valid_N,
                                    num_cores=4,
                                    tokenizer=bert_tokenizer,
                                    max_length=MAX_LENGTH)

print("Validation data encoding complete.")


test_input_ids, test_token_type_ids, test_attention_mask \
                        = test_encoding_process(
                                    df=test_df,
                                    N=test_N,
                                    num_cores=4,
                                    tokenizer=bert_tokenizer,
                                    max_length=MAX_LENGTH)

print("Test data encoding complete.")

save_data(compressed=True)

x_train = np.load('train-df-compressed-input-ids.npz', mmap_mode='r')
x_train.f.arr_0.shape
from multiprocessing import Pool

num_cores = 4

def fx(t):
    t1, t2 = t
    for i in tqdm(range(t1, t2)):
        z = i * i
        #print(f"{z}")


def process():
        
    n1 = (0, int(N/4))
    n2 = (int(N/4), int(N/2))
    n3 = (int(N/2), int(0.75 * N))
    n4 = (int(0.75 * N), N)
    
    n = [n1, n2, n3, n4]
    
    pool = Pool(num_cores)
    result = pool.map(fx, n)
    return result
    
ps = process()
def fast_batch_encoding(texts, tokenizer, batch_size, max_len=192):
    input_ids = []
    for i in tqdm(range(0, len(texts), batch_size)):
        text = texts[i: i + batch_size].tolist()
        inputs = tokenizer.batch_encode_plus(text, 
                                             pad_to_max_length=True, 
                                             max_length=max_len)
        input_ids.extend(inputs['input_ids'])
    return np.array(input_ids)

input_ids = fast_batch_encoding(train_df.comment_text.values, tokenizer, batch_size=256)
input_ids.shape
tokenizer.decode(list(input_ids[0]))