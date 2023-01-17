# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

# -*- coding: utf-8 -*-



import xgboost as xgb



def loaddata(address):

    csv_data = pd.read_csv(address) 

    csv_data = csv_data.fillna(0)

    data = np.array(csv_data)

    length = data.shape[0] 

    label = data[0:length,1]

    data = data[0:length,2:12]

    return label,data



[label,data] = loaddata('/kaggle/input/GiveMeSomeCredit/cs-training.csv')

[new_label,new_data] = loaddata('/kaggle/input/GiveMeSomeCredit/cs-test.csv')



params = {

    'booster':'gbtree', 

    'slient':1,

    

    'objective':'reg:logistic',

    'eval_metric':'auc',

    'seed':1000,  

     

    'eta':0.1,

    'gamma':0.1,

    'lambda':3,

    'max_depth':8,

    'subsample':0.7,

    'colsample_bytree':0.7,

    'min_child_weight':3,

    

    }

nums_rounds = 300





dtrain = xgb.DMatrix(data,label)

dtest = xgb.DMatrix(new_data)



model = xgb.train(params,dtrain,nums_rounds)

new_label = model.predict(dtest)



length = len(new_label)

ind = np.arange(1,length+1)

df = pd.DataFrame(ind,columns = ['Id'])

df['Probability'] = pd.DataFrame(new_label)



export_csv = df.to_csv('export_dataframe.csv',index = None, header = True)
