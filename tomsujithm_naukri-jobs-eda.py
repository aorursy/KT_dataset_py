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
import pandas as pd
df = pd.read_csv("../input/naukri-dataset/naukri.csv")

df.head()
df.shape
df.info()
df.isnull().sum().sort_values(ascending=False)
round(100*df.isnull().sum()/len(df),2).sort_values(ascending=False)
import seaborn as sns

sns.heatmap(df.isnull(),yticklabels=False)
df.dropna(inplace=True)

df.shape
df['Crawl Timestamp'].min()
df['Crawl Timestamp'].max()
df['Job Salary'].value_counts()
df['Key Skills'].value_counts()
df['Role Category'].value_counts()
df['Location'].value_counts()
df['Functional Area'].value_counts()
df['Industry'].value_counts()
df['Role'].value_counts()
df.drop(['Uniq Id'],inplace = True,axis=1)
df.rename(columns = {'Job Experience Required':'Experience'}, inplace=True)
df['Experience'].replace('30 years and above','30 - 50 Yrs',inplace = True)
df['Experience'] = df['Experience'].str.strip()
df['Min_Exp'] = df.Experience.str.split("-",expand=True,)[0]
df['Max_Exp'] = df.Experience.str.split("-",expand=True,)[1]
df['Max_Exp'] = df.Max_Exp.str.extract('(\d+)')
df.head()