# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.shape)
print(test.shape)
train.columns
print(check_output(["head", "../input"]).decode("utf8"))
submission = pd.DataFrame()
submission["Id"] = test.Id

def loadData(df, test = None):
    
    dt = pd.to_datetime(df.Dates).dt
    df["Year"] = dt.year
    df["Month"] = dt.month
    df["Day"] = dt.day
    df["Hour"] = dt.hour
    #df["Minute"] = df.minite
    df["Week"] = dt.week
    df.drop("Dates", axis = 1, inplace = True)
    
    df["AddressIsOf"]= df.Address.str.contains('.?of.?')
    df.drop("AddressIsOf", axis = 1, inplace = True)
    
    if test:
        df.drop("Id", axis = 1, inplace = True)
        y = None
    else:
        df.drop("Descript", axis = 1, inplace = True)
        df.drop("Resolution", axis = 1, inplace = True)
        y = df.Category
        df.drop("Category", axis = 1, inplace = True)
        
    X = df
    
    return X, y

X, y = loadData(train)
X
### small test here
pd.to_datetime(test.Dates).dt.week
test.Dates.tail()

#grep('.?of.?', test.Address)

mytest = test.copy()
mytest["AddressIsOf"]= mytest.Address.str.contains('.?of.?')
mytest.head()
# setup parameters for xgboost
param = {}
# use logistic regression loss, use raw prediction before logistic transformation
# since we only need the rank
param['booster'] = 'gbtree'
param['objective'] = 'multi:softprob'
# scale weight of positive examples
#param['scale_pos_weight'] = sum_wneg/sum_wpos
param['eta'] = 1.0
#param['num_class'] = m
param['max_depth'] = 6
param['max_delta_step'] = 1
#param['silent'] = 1
#param['nthread'] = 16

num_round = 10
bst = xgb.train( param.items(), dtrain, num_round, evallist )
X.head()