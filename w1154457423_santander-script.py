
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
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
remove=[]
col=train.columns
for i in len(col)-1:
    k=train[c[i]]
    for j in range(i+1,len(c)):
        k1=train[c[j]]
        if np.array_equal(k,k1):
            remove.append(c[j])
train.drop(remove,axis=1)
test.drop(remove,aixs=1)