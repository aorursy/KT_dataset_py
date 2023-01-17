# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import rcParams

pd.set_option('display.max_colwidth', -1)

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/the-city-of-new-york-jobs/NYC_Jobs.csv")
df.head()
df.isna().sum()
rcParams['figure.figsize'] = 15,10

df["Job Category"].value_counts()[:10].plot(kind="pie")
rcParams['figure.figsize'] = 15,10

df["Business Title"].value_counts()[:20].plot(kind="bar",colormap="Set2")
rcParams["figure.figsize"] = 15,20

df["Work Location"].value_counts()[:20].plot(kind="pie")
df["Agency"].value_counts().plot(kind="bar")
sns.countplot(x=df["Posting Type"],hue=df["Title Classification"])