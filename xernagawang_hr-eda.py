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
import numpy as np 

import pandas as pd 



import time 
df = pd.read_csv("../input/human-resources-data-set/core_dataset.csv")

display(df.shape)

display(df.head())
display(df.columns)
df['DOB'] = pd.to_datetime(df['DOB'])
df["Date of Hire"] = pd.to_datetime(df["Date of Hire"])
df.shape
import seaborn as sns
m=[i.month for i in df["Date of Hire"]]
df_m = pd.DataFrame(m)

df_m.columns = ["month"]

df_m = df_m["month"]

df_m.value_counts().plot(kind = "bar", figsize = (10, 8))
df["Sex"].value_counts()
# 首先去掉所有还在公司的数据



df_tr = df[df["Reason For Term"] != "N/A - still employed"]

df_tr["Reason For Term"].value_counts().plot(kind = "bar", figsize = (10, 8))
pd.pivot_table(df,index=["Reason For Term"])
df["Department"].value_counts().plot(kind = "bar", figsize = (10, 8))
pd.pivot_table(df,index=["Department"])
df["Employee Source"].value_counts().plot(kind = "bar", figsize = (10, 8))