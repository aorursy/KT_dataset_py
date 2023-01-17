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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



pd.set_option('display.max_columns',None)
df = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

df.head()
df.drop('sl_no',axis=1,inplace=True)
df.head()
df.info()
df.shape
sns.countplot(df['gender'])
df['status'].value_counts()
sns.countplot(df['status'])
print(np.round(148/215,4)*100,'% of students got placed.')
print(df['workex'].value_counts())

sns.countplot(df['workex'])
print(np.round(141/215,4)*100,'% of students had work experience.')
print(f'{len(df.loc[(df.status=="Placed") & (df.workex=="Yes")])} students got placed and already had some work experience.')
print(f'{len(df.loc[(df.status=="Placed") & (df.degree_p<70)])} students got placed with a degree percentage below 70%.')
df.columns
sns.countplot(df['specialisation'])
sns.countplot(df['degree_t'])
print('Commerce and Management students are dominating.')
marks = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']

for col in marks:

    sns.distplot(df[col])

    plt.show()
sns.countplot(df['hsc_s'])