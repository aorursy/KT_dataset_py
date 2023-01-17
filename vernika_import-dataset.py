import numpy as np 

import pandas as pd 

import os



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv', index_col = 0)
train_data.head()
train_data.shape
train_data.info()