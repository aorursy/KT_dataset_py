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
df = pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1', header=None)
dfn = pd.read_csv('/kaggle/input/sanders/full-corpus.csv', header=None)
label_text = df.iloc[:, [5, 0]]
label_text[0] = label_text[0].apply(lambda x: int(x / 2))
label_text[5] = label_text[5].apply(lambda x: x.strip())
label_text = label_text.drop_duplicates(subset=5, keep='first')

# col_list = list(label_text)
# col_list[0], col_list[1] = col_list[1], col_list[0]
# label_text.columns = col_list
label_text.info
pos = label_text[label_text[0] == 2][:3833]
neg = label_text[label_text[0] == 0][:3833]
pos.columns= ['text', 'label']
neg.columns= ['text', 'label']
pos.head()
# label_text_10k = pd.concat([pos, neg])
# label_text_10k.info
from os import listdir
from os.path import isfile, join
mypath = '/kaggle/'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)
label_text_n = dfn.iloc[:, [4, 1]]
label_text_n[1] = label_text_n[1].apply(lambda x: 2 if x == 'positive' else (1 if x == 'neutral' else (0 if x == 'negative' else 3)) )
label_text_n[4] = label_text_n[4].apply(lambda x: x.strip())
label_text_n = label_text_n.drop_duplicates(subset=4, keep='first')
label_text_n.info
neutral = label_text_n[label_text_n[1] == 1]
neutral.columns= ['text', 'label']
neutral.info
label_text_10k_3classes = pd.concat([pos, neg, neutral])
label_text_10k_3classes[label_text_10k_3classes['label'] == 1].info
label_text_10k_3classes.to_csv('/kaggle/working/label_text_10k_3classes.csv', index=False, encoding='utf-8', header=False)