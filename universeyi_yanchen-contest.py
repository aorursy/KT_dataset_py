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
%matplotlib inline

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import scipy.stats as stats

import sklearn.linear_model as linear_model

import seaborn as sns

import xgboost as xgb

from sklearn.model_selection import KFold

from IPython.display import HTML, display

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



pd.options.display.max_rows = 1000

pd.options.display.max_columns = 20



train = pd.read_csv('../input/new yancheng_train_20171226.csv',na_values='-',keep_default_na=False)

#train = pd.read_csv('../input/new yancheng_train_20171226.csv')

test = pd.read_csv('../input/yancheng_testA_20171225.csv')





quantitative = [f for f in train.columns if train.dtypes[f] != 'object']

qualitative = [f for f in train.columns if train.dtypes[f] == 'object']



#print(quantitative)

print(qualitative)



#train['level_id'].sample(20)

#print(train[qualitative].head())




train_sum10=train[(train.sale_date==201710)].groupby(['class_id']).sale_quantity.sum().round()



predicted=train_sum10.reset_index()



result=pd.merge(test[['predict_date','class_id']],predicted,how='left',on=['class_id'])



result.fillna(0)



result.columns=['predict_date','class_id','predict_quantity']

result['predict_quantity'].apply(lambda x:x *1.2)

   



result.to_csv('result_201710_1.2.csv',index=False,header=True)  