% ls ../input
import pandas as pd

import missingno as msno

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
target_col = "SalePrice"

target = train[target_col]

train.drop(columns=[target_col], inplace=True)
mix = pd.concat([train, test])
print("train.shape: {}".format(train.shape))

print("test.shape: {}".format(test.shape))

print("mix.shape: {}".format(mix.shape))
train.head()
test.head()
msno.matrix(train, labels=list(train.columns))
msno.matrix(test, labels=list(test.columns))
msno.matrix(mix, labels=list(mix.columns))
msno.bar(train)
msno.bar(test)
msno.bar(mix)
msno.heatmap(train)
msno.heatmap(test)
msno.heatmap(mix)