##### Global Imports

import pandas as pd

import random

# Filter Row Numbers



n = 125497040 #number of records in file (excludes header)

s = 10000 #desired sample size

select = sorted(random.sample(range(1,n+1),s))

skip  = tuple(set(range(1,n+1)) - set(select))

df_train = pd.read_csv("../input/train.csv",skiprows=skip)
df_train.shape