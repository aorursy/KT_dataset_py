# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
DATADIR = "../input"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
# Any results you write to the current directory are saved as output.

import ipywidgets as wg
from IPython.display import display
import os
train = pd.read_csv(os.path.join(DATADIR, TRAIN_FILE))

test = pd.read_csv(os.path.join(DATADIR, TEST_FILE))
def toxic_cookie(level="toxic"):
    print(train.iloc[np.random.choice(np.where(train[level] == 1)[0]),1])
    print("-"*40)
    print(train.iloc[np.random.choice(np.where(train[level] == 1)[0]),[0,2,3,4,5,6,7]])
wg.interact_manual(toxic_cookie, level=list(train.columns[2:].values))