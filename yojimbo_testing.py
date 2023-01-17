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

data = pd.read_csv('../input/train.csv')
data.head(2)
#print ('done')



train = [
    [-1000,0,.5,2,4],
    [3,2,1,2,1],
    [4,1,1,2,7],
    [500,1,0,2,5]]

from sklearn import preprocessing
sc = preprocessing

t = sc.StandardScaler().fit(train)

r = t.transform(train)

r



