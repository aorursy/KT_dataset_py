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
import pandas as pd

import numpy as np

cbp=pd.read_table("../input/comp-score/CB1600CZ11.dat", sep="|")

cbp=cbp.loc[:, ["ZIPCODE", "ESTAB"]]
train=pd.read_csv("../input/comp-score/train.csv")

test=pd.read_csv("../input/comp-score/test.csv")

train=train.merge(cbp, how="left", left_on="zipcode", right_on="ZIPCODE")

test=test.merge(cbp, how="left", left_on="zipcode", right_on="ZIPCODE")
# Observe ESTAB by group

observe=train.groupby(by="score")["ESTAB"].agg([np.min, np.max])

observe.reset_index(inplace=True)

observe
train["generate"]=train["ESTAB"].apply(lambda x: int((x/14+0.9999)/2))

from sklearn import metrics

np.sqrt(metrics.mean_squared_error(train["score"], train["generate"]))
# train["dif"]=train["score"]-train["generate"]

# train[train["dif"]!=0]

observe["floor"]=observe["score"]*28-13

observe["ceil"]=observe["score"]*28+14

abnormal=observe[(observe["amin"]<observe["floor"]) | (observe["amax"]>observe["ceil"])]

abnormal["dif"]=abnormal["ceil"]-abnormal["amax"]

abnormal
train["generate"]=train["ESTAB"].apply(lambda x: int((x/(14+1/106)+0.9999)/2))

np.sqrt(metrics.mean_squared_error(train["score"], train["generate"]))

# test["score"]=test["ESTAB"].apply(lambda x: int((x/(14+1/106)+0.9999)/2))

# test.loc[:, ["id", "score"]].to_csv("submit.csv")