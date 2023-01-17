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
import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

train = pd.read_csv(os.path.join('../input', 'train.csv'))

test = pd.read_csv(os.path.join('../input', 'test.csv'))
train.info()
train.head()
train['Survived'].value_counts(normalize=True)
sns.countplot(train['Survived'])
train['Survived'].groupby(train['Pclass']).mean()

sns.countplot(train['Pclass'],hue=train['Survived'])
train['Name'].head()
train['Name_Title']=train['Name'].apply(lambda x:x.split(',')[1]).apply(lambda x:x.split()[0])
train['Name_Title'].value_counts()
train['Survived'].groupby(train['Name_Title']).mean()
train['Name_len']=train['Name'].apply(lambda x:len(x))