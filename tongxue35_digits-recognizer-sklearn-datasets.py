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
from sklearn import datasets

iris=datasets.load_iris()

digits=datasets.load_digits()

print(len(digits.data[:1][0]),len(digits.data),digits.data.shape)

#可以看到data64个维度，1797样本
# 数据原来格式为8*8

digits.images[0]

# 分别转换为1*n  N*1数组

# digits.images[0].reshape(1,-1)

# digits.images[0].reshape(-1,1)
from sklearn import svm

from sklearn.model_selection import train_test_split

#将数据分为训练集和检验集

train_set,verify_set,train_label,verify_label=train_test_split(digits.data,digits.target,test_size=0.3,random_state=20)

# print(train_set.shape,verify_set.shape)

clf=svm.SVC(gamma=0.001,C=100.)

clf.fit(train_set,train_label)
predict=clf.predict(verify_set)

df=pd.DataFrame({'label':verify_label,"predict":predict})

# df.label[df['label']==df['predict']].count()/len(df)

#预测结果100%

clf.score(verify_set,verify_label)

# 还可以做降维提升数据的质量，节约训练时间
