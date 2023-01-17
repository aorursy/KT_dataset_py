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
import matplotlib.pyplot as plt
import seaborn as sns
dat=pd.read_csv('../input/NBA_train.csv')
dat.info()
dat.head()
dat1=dat.loc[:,['PTS','oppPTS']]
sns.jointplot(dat1.PTS,dat1.oppPTS,kind='reg',size=(8))
dat2=dat.loc[:,['PTS','oppPTS','FTA','ORB','AST','STL','BLK','TOV']]
sns.pairplot(dat2,size=(3))
dat.groupby(by=['Team','SeasonEnd']).agg({'W':np.sum}).unstack().head(5)



