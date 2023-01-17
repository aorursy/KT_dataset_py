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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/../input/google-play-store-apps/googleplaystore.csv')
print(df.shape)
print(df.columns)
print(df.dtypes)
df.head()
df.tail()
df.info()
print(df.isnull().sum().sort_values(ascending=False))
print(df.describe())
print(df["Category"].describe())
df.describe(include=['O'])
df["Category"].value_counts()
df[df["Category"]=="1.9"]
df.loc[10472,'Category'] = "POTOGRAPHY"
df.sort_values(by = 'Rating', ascending = False).head(10)
df = df.drop(10472)


df.sort_values(by='Rating', ascending=False).head(10)
df= df.drop("Content Rating", axis=1)
print(df.columns)
df= df.drop('Content Rating', axis=1)
print(df.columns)
print(df.groupby("Category")["Rating"].mean().head(10))
print(df.groupby(["Category", "Type"]) ["Rating"].max().head(8))
df["Rating"].fillna(df.groupby("Category") ["Rating"].transform("mean"), inplace = True)
print(df.isnull().sum().sort_values(ascending=False))
df = df.dropna()
print(df.isnull().sum().sort_values(ascending=False))