# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5)
df=pd.read_csv('../input/train.csv')
df2=pd.read_csv('../input/test.csv')
testdat=np.array(df2)
Y=np.array(df.iloc[:,0])
X=np.array(df.iloc[:,1:])
clf.fit(X, Y)
pred = clf.predict(testdat)
print(pred)
dff = pd.DataFrame(pred)
dff.to_csv('ans.csv')



# Any results you write to the current directory are saved as output.
