!pip install textgenrnn

import pandas as pd

import re

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers import Dense, Activation,LSTM, Dense, Dropout, Embedding

from keras.callbacks.callbacks import ModelCheckpoint

import numpy as np

from keras.utils import np_utils

from keras.preprocessing.sequence import pad_sequences

from textgenrnn import textgenrnn

import os
os.listdir("/kaggle/input/")
video_path = "/kaggle/input/uscomment/USvideos.csv"

comment_path="/kaggle/input/uscomment/UScomment.csv"

txt_path = "/kaggle/input/data-txt/train.txt"

weight_path = "/kaggle/input/weight/hacker_news.hdf5"

context_path = "/kaggle/input/context/context_example.csv"
video = pd.read_csv(video_path)

comment = pd.read_csv(comment_path,header=None)

merged = pd.merge(left=video, right=comment, left_on='video_id', right_on=0)

merged = merged[merged[1].notna()]
merged.head()
all_txt = list(merged[1].astype(str).values)

all_label = list(merged["video_id"].astype(str).values)
train_num = 50000

to_train_txt = all_txt[:train_num]

to_train_label = all_label[:train_num]
to_train_txt[:1000]
textgen = textgenrnn(weight_path)
textgen.train_on_texts(to_train_txt,num_epochs=10,gen_epochs=3,batch_size=128)
textgen.save("model3.hdf5")
def generate(model,max_len=30,p=None):

    r = model.generate(top_n=2,max_gen_length=70,prefix=p,temperature=0.6,return_as_list=True)



    while(len(r[0])>=max_len):

        r = model.generate(top_n=2,max_gen_length=70,prefix=p,temperature=0.6,return_as_list=True)

    return r[0]

    

    

    

    



    
for i in range(3):

    r = generate(textgen,p="what")

    print(r)