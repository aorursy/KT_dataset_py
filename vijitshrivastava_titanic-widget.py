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
train=pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
train['Salutation']=train['Name'].map(lambda x : x.split(',')[1].split('.')[0])

train.head()
p=train['Salutation'].value_counts()

p
def collect_rare(data,attr,rare):

    p=train['Salutation'].value_counts()

    for k,v in data[attr].items():

        if(p[v]<10):

            data[attr][k]=rare

    return data

collect_rare(train,'Salutation','Rare')

train['Salutation'].value_counts()

train.describe()
train.head()
train.describe()
print(train['Age'] & (train['SibSp'][i]>=2))
