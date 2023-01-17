# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import time



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DATA_PATH = "../input/jigsaw-multilingual-toxic-comment-classification"

small_path = "jigsaw-toxic-comment-train.csv"

large_path = "jigsaw-unintended-bias-train.csv"

val_path = "validation.csv"

test_path = "test.csv"

SEQUENCE_LENGTH = 192
small_ds = pd.read_csv(os.path.join(DATA_PATH, small_path), usecols=["comment_text", "toxic"])

large_ds = pd.read_csv(os.path.join(DATA_PATH, large_path), usecols=["comment_text", "toxic"])

aug_ds = pd.read_csv("../input/class-balancing/aug.csv")

val_ds = pd.read_csv(os.path.join(DATA_PATH, val_path), usecols=["comment_text", "toxic"])

test_ds = pd.read_csv(os.path.join(DATA_PATH, test_path))
aug_ds = aug_ds.sample(300000)
large_toxic = large_ds[large_ds["toxic"] > 0.5].round()

large_nontoxic = large_ds[large_ds["toxic"] == 0].sample(600000)



ds = pd.concat((small_ds,

              large_toxic,

              large_nontoxic,

              aug_ds))
len(ds)
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\xa0', '\t',

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
def clean_text(text, lang='en'):

    text = str(text)

    text = re.sub(r'[0-9"]', '', text)

    text = re.sub(r'#[\S]+\b', '', text)

    text = re.sub(r'@[\S]+\b', '', text)

    text = re.sub(r'https?\S+', '', text)

    text = re.sub(r'\s+', ' ', text)

    text = re.sub("\[\[User.*",'',text)

    for punct in puncts:

        text = text.replace(punct, "")

    return text



def clean_data(df, text_label="comment_text", train=True):

    pos = 0

    while pos < len(df):

        temp = df[pos:pos + 10000].copy()

        df[pos:pos+10000][text_label] = temp[text_label].apply(clean_text).values

        pos += 10000

        print("Processed", pos, "texts" )

    df["lens"] = df[text_label].str.split().apply(len)

    if train:

        df = df[df["lens"] > 0]

    df.drop("lens", axis=1, inplace=True)

    return df
cleaned_ds = clean_data(ds)

cleaned_val = clean_data(val_ds)

cleaned_test = clean_data(test_ds,text_label="content", train=False)
len(cleaned_ds) == len(test_ds)
from transformers import AutoTokenizer



MODEL = "xlm-roberta-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
def encode_comments(dataframe, tokenizer=tokenizer, max_len=SEQUENCE_LENGTH):

        

        pos = 0

        start = time.time()

        

        while pos < len(dataframe):

            temp = dataframe[pos:pos+10000].copy()

            res = tokenizer.batch_encode_plus(temp.comment_text.values,

                                              pad_to_max_length=True,

                                              max_length = SEQUENCE_LENGTH,

                                              return_attention_masks = False

                                             )

            if pos == 0:

                ids = np.array(res["input_ids"])

                labels = temp.toxic.values

            else:

                ids = np.concatenate((ids, np.array(res["input_ids"])))

                labels = np.concatenate((labels, temp.toxic.values))

            pos+=10000

            print("Processed", pos, "elements")

        return ids, labels
ids,labels = encode_comments(cleaned_ds)

val_ids,val_labels = encode_comments(cleaned_val)
test_ids = tokenizer.batch_encode_plus(cleaned_test.content.values,

                                      pad_to_max_length=True,

                                      max_length=SEQUENCE_LENGTH,

                                      return_attention_masks=False)["input_ids"]
#save the results of tokenization for future use in the next notebooks



np.save("ids.npy", ids)

np.save("labels.npy", labels)

np.save("val_ids.npy", val_ids)

np.save("val_labels.npy", val_labels)

np.save("test_ids.npy", test_ids)