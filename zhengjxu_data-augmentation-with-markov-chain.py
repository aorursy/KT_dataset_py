# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import markovify as mk
from joblib import Parallel, delayed
train = pd.read_csv('../input/train.csv')
insincere = train.loc[train.target==1, ['question_text', 'target']]
insincere.head()
nchar = int(insincere.question_text.str.len().median())
nchar
text_model = mk.Text(insincere['question_text'].tolist())
def data_augment():
    return text_model.make_short_sentence(nchar)
parallel = Parallel(-1, backend="threading", verbose=5)
count = 1000
aug_data = parallel(delayed(data_augment)() for _ in range(count))
aug_data[:5]