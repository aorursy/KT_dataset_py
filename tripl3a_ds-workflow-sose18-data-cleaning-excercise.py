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
df = pd.read_csv('../input/dsm-beuth-edl-demodata-dirty.csv')
df
df = df.drop_duplicates(["full_name","first_name","last_name","email","gender","age"])
df
df = df.dropna(how="all") 
df
if pd.isnull(df.loc[21,"id"]): 
    df.at[21,"id"] = max(df.id)+1
df.tail()
df["age"] = pd.to_numeric(df.age, errors="coerce")
df
invalid_age_idx = df.age<0
print(df[invalid_age_idx][["id","age"]])
df.loc[invalid_age_idx,"age"] = None
print(df[invalid_age_idx][["id","age"]])
df.at[5, "gender"] = "Male"
df.loc[5]
df