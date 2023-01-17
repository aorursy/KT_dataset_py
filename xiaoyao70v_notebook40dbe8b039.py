import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']), ignore_index=True)