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
train = pd.read_csv('../input/train.csv')

train.head()
import sklearn

from sklearn.decomposition import FastICA, KernelPCA

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline
train['labs'] = LabelEncoder().fit_transform(train['type'])
cols = ['bone_length','rotting_flesh', 'hair_length','has_soul']

demix = Pipeline([FastICA(n_components=4),KernelPCA(n_components=2)])

mat = demix.transform(train[cols])

print(mat.shape)
import matplotlib.pyplot as plt

%matplotlib inline

colors=[(255,0,0),(0,255,0),(0,0,255)]

plt.scatter(mat[:,0],mat[:,1],c=train['labs'])

 