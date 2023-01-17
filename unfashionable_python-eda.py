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
train = pd.read_csv("../input/data_set_ALL_AML_train.csv")
train.shape
train.head()
train = train[[c for c in train.columns if "call" not in c]]
train.head()
test = pd.read_csv("../input/data_set_ALL_AML_independent.csv")
test = test[[c for c in test.columns if "call" not in c]]
test.head()
train = train.set_index('Gene Accession Number')


test = test.set_index('Gene Accession Number')
train.head()
data = pd.concat([train,test],axis=1)
data.head()
data = data.drop('Gene Description',axis=1)
data.head()
data = data.T
data.head()
data.index
label = pd.read_csv("../input/actual.csv")
label.head()
label = label.set_index('patient')
temp = label.to_dict()

temp.keys()
temp['cancer'][1]
data['class'] = data.index
data.head()
data['class'] = data['class'].astype(int)
data['class'].replace(temp['cancer'],inplace=True)

data['class'].replace({'ALL':0,'AML':1},inplace=True)
data.head()
data['class'].plot(kind='hist')
import seaborn as sns
diff = data.groupby('class').mean().apply(lambda x: x[0]-x[1])
my_columns = diff.sort_values().index.tolist()
selected = my_columns[:100] + my_columns[-100:]
len(selected)
small_data = data[selected + ['class']]
small_data.shape
sns.set(font_scale=0.5)
sns.clustermap(small_data.corr(),cmap='RdBu_r',figsize=(15,15))
sns.clustermap(small_data,standard_scale=1,method="ward")

















