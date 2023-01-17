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

import seaborn as sns

import matplotlib.pyplot as plt

% matplotlib inline

import tensorflow as tf

tf.reset_default_graph()

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.contrib import learn

from sklearn import cross_validation

from sklearn import preprocessing

from sklearn import metrics

from __future__ import print_function



%matplotlib inline
df1=pd.read_csv('../input/data_set_ALL_AML_independent.csv')

df2=pd.read_csv('../input/data_set_ALL_AML_train.csv')
print(df1.columns)

print(df2.columns)
print(df1.head())

print(df2.head())
df1.replace(['A','P','M'],['1','2','3'],inplace=True)

df2.replace(['A','P','M'],['1','2','3'],inplace=True)
df1
sns.heatmap(df1.corr()) 

plt.figure(figsize=(18,12))  



 

sns.heatmap(df2.corr()) 