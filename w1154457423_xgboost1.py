# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import ensemble
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score as auc
import time

train=pd.read_csv('../input/train.csv')
colstoremove=[]
columns=train.columns
for i in range(len(columns)-1):
    v=train[columns[i]].values
    for j in range(i+1,len(columns)-1):
        if np.array_equal(v,train[columns[j]]):
            colstoremove.append(columns[j])
train.drop(colstoremove,axis=1,inplace=True)
trainlabels=train.TARGET
trainfeature=train.drop(['ID','TARGET'],axis=1)

x_train,y_vail,y_train,y_vail=cross_validation.train_test_split(trainfeature,trainlabels,test_size=0.5,random_state=1)