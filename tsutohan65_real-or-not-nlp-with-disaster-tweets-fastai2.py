# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
! /opt/conda/bin/python3.7 -m pip install --upgrade pip
# 下記コマンドではfastai v2 をインストールできない。
#! conda install --yes -c fastai -c pytorch -c conda-forge fastai=2.0.13=py_0
# 最新バージョンを確認する
#! conda search -c fastai --full-name fastai
# https://www.kaggle.com/vishynair/fastai-2-0-13-installation-with-kaggle-notebook
!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# pytorch のインストールで kornia と allennlp のアップデートが必要というエラーメッセージが表示された場合、以下を実行する。
!pip install --upgrade kornia
!pip install allennlp==1.1.0.rc4
!pip install --upgrade fastai
from fastai.text.all import *
df = pd.read_csv(Path('/kaggle/input/nlp-getting-started/train.csv'))
df
dt = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
dt
data_lm = TextDataLoaders.from_df(df, text_col='text', is_lm=True)
data_lm.show_batch(max_n=3)
data_lm.vocab[:10]
data_lm.numericalize
data_lm.tokenizer
data_lm.rules
def print_rules(dls):
    "Prints out current rules of `Tokenizer`"
    print(f"{dls.tokenizer.__doc__} with the following rules\n")
    [print(f"{r.__name__, r.__doc__}") for r in dls.rules]
print_rules(data_lm)
data_lm.o2i.items()
learn = language_model_learner(
    data_lm, AWD_LSTM, drop_mult=0.3, 
    metrics=[accuracy, Perplexity()]).to_fp16()
learn.fit_one_cycle(1, 2e-2)
learn.save('1epoch')
learn.unfreeze()
learn.fit_one_cycle(10, 2e-3)
# 最終アクティベーションを含まないモデルを保存する
learn.save_encoder('finetuned')
blocks = (TextBlock.from_df('text', seq_len=data_lm.seq_len, vocab=data_lm.vocab), CategoryBlock())
dls_clas = DataBlock(
    blocks=blocks,
    get_x=ColReader('text'),
    get_y=ColReader('target'),
    splitter=RandomSplitter()
)
dls = dls_clas.dataloaders(df, bs=64)
dls.show_batch(max_n=3)
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy).to_fp16()
learn.load_encoder('finetuned')
learn.fit_one_cycle(1, 2e-2)
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2))
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))
learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3))
learn.save('classifier')
preds = [[row['id'], learn.predict(row['text'])[0]] for index, row in dt.iterrows()]
preds[:10]
import csv

with open("submission.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["id", "target"])
    writer.writerows(preds)