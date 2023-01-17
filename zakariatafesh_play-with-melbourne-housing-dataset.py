# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

df
df.columns
df.isnull().any()
df.YearBuilt
df.YearBuilt.isnull()
any([True, True, True]), any([True, False, True]), any([False, False, False])
all([True, True, True]), all([True, False, True]), all([False, False, False])
df.isnull().sum()
df.shape
for col in df.columns:

    if df[col].isnull().any():

        print(col, df[col].isnull().sum())
for col in df.columns:

    if df[col].dtype == 'object':

        print(col)
for col in df.columns:

    if df[col].dtype == 'int64' or df[col].dtype == 'float64':

        print(col)
# Get list of categorical variables

s = (df.dtypes == 'object')

print(s)

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
df_num = df.select_dtypes(exclude='object')

df_num
df_cat = df[object_cols]

df_cat