# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv') # Read the first few rows

df.head()
numeric_features = df.dtypes[df.dtypes != 'object']

num_df = df[numeric_features.index].astype(float)

num_df.drop('Id',axis=1,inplace=True)

num_df.fillna(num_df.mean())

num_df.head()
label = num_df["SalePrice"].astype(float)

num_df.drop(["SalePrice"],axis=1,inplace=True)

label.hist(bins=30,alpha=0.3,color='k')
names = []

corr = []

for col in num_df.columns:

    names.append(col)

    corr.append(np.corrcoef(num_df[col].values, label.values)[0,1])

relation = pd.DataFrame(data=corr, index=names, columns=["corr"])

relation.head(20)
relation.plot(kind='barh',figsize=(6, 15))