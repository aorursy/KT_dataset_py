# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', 500)

pd.set_option('display.max_rows', 1000)

import time

from sklearn.metrics import log_loss



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def str_to_num(string):

    return int(string.split(" ")[1])



train=pd.read_csv('../input/train.csv', converters={'location':str_to_num})



sample=train.loc[51:55]

sample
true=np.array([2,1,1,0,2])

pred=np.array([[.1, .1, .99], [.1, .7, .1], [.2, .7, .1], [.8, .1, .1], [.1, .1, .6]])
logloss=-(1*np.log(0.99)+1*np.log(0.7)+1*np.log(0.7)+1*np.log(0.8)+1*np.log(0.6))/5

print('hand calculated logloss:', logloss)
logloss=-(1*np.log(0.99)+1*np.log(0.7)+1*np.log(0.7)+1*np.log(0.8)+1*np.log(0.6))/5

print('hand calculated logloss:', logloss)



sklearn_logloss=log_loss(true, pred)

print('sklearn_logloss:', sklearn_logloss)
# normalisation of prediction 

pred_norm=pred/pred.sum(axis=1)[:, np.newaxis]

pred_norm
norm_log_loss=-(1*np.log(0.832)+1*np.log(0.778)+1*np.log(0.7)+1*np.log(0.8)+1*np.log(0.75))/5

print('logloss:', logloss)

print('norm_log_loss:', norm_log_loss)

print('sklearn_logloss:', sklearn_logloss)