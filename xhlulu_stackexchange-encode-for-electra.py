!pip -q install transformers --upgrade

!pip show transformers
import os



import numpy as np

import pandas as pd

import transformers as trfm

from tokenizers import BertWordPieceTokenizer

from tqdm.notebook import tqdm
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512, enable_padding=False):

    """

    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras

    

    ---

    

    Inputs:

        tokenizer: the `fast_tokenizer` that we imported from the tokenizers library

    """

    tokenizer.enable_truncation(max_length=maxlen)

    if enable_padding:

        tokenizer.enable_padding(max_length=maxlen)

    

    all_ids = []

    

    for i in tqdm(range(0, len(texts), chunk_size)):

        text_chunk = texts[i:i+chunk_size].tolist()

        encs = tokenizer.encode_batch(text_chunk)

        all_ids.extend([enc.ids for enc in encs])

    

    return np.array(all_ids)
def combine_qa_ids(q_ids, a_ids, tokenizer, maxlen=512):

    """

    Given two arrays of IDs (questions and answers) created by

    `fast_encode`, we combine and pad them.

    Inputs:

        tokenizer: The original tokenizer (not the fast_tokenizer)

    """

    combined_ids = []



    for i in tqdm(range(q_ids.shape[0])):

        ids = []

        ids.append(tokenizer.cls_token_id)

        ids.extend(q_ids[i])

        ids.append(tokenizer.sep_token_id)

        ids.extend(a_ids[i])

        ids.append(tokenizer.sep_token_id)

        ids.extend([tokenizer.pad_token_id] * (maxlen - len(ids)))



        combined_ids.append(ids)

    

    return np.array(combined_ids)
df = pd.concat([

    pd.read_csv(f"/kaggle/input/stackexchange-qa-pairs/pre_covid/{group}.csv")

    for group in ['general', 'expert', 'biomedical']

])



questions = df.title + ' [SEP] ' + df.question
tokenizer = trfm.ElectraTokenizer.from_pretrained("google/electra-small-discriminator")

# Save the loaded tokenizer locally

tokenizer.save_pretrained('.')



# Reload it with the huggingface tokenizers library

MAX_LEN = 512

fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=True, add_special_tokens=False)
q_ids = fast_encode(questions.values, fast_tokenizer, maxlen=MAX_LEN//2 - 2)

a_ids = fast_encode(df.answer.values, fast_tokenizer, maxlen=MAX_LEN//2 - 2)

wa_ids = fast_encode(df.wrong_answer.values, fast_tokenizer, maxlen=MAX_LEN//2 - 2)
correct_ids = combine_qa_ids(q_ids, a_ids, tokenizer, maxlen=MAX_LEN)

wrong_ids = combine_qa_ids(q_ids, wa_ids, tokenizer, maxlen=MAX_LEN)
np.save("correct_ids.npy", correct_ids)

np.save("wrong_ids.npy", wrong_ids)