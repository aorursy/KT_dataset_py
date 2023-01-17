import pandas as pd

import numpy as np

import seaborn as sbn

import matplotlib.pyplot as plt
train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")

predCol = train['type']

data = pd.concat([train.drop('type',axis=1),test],axis=0)

data.head()
sbn.pairplot(data.drop('id',axis=1))
g = sbn.FacetGrid(train,col="type")

g.map(plt.hist,'has_soul')