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
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
print("all libraries are imported")
df=pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df=pd.DataFrame(df)
df.sample(5)
df.info()
s=df.shape
print("There are {} number of rows.".format(s[0]))
print("There are {} number of columns.".format(s[1]))
np.sum(pd.isnull(df))
df['salary'] = df['salary'].replace(np.nan, 0)

np.sum(pd.isnull(df['salary']))
df.columns
df1 = df
df1['status'].values[df1['status']=='Not Placed'] = 0 
df1['status'].values[df1['status']=='Placed'] = 1
df1.status = df1.status.astype('int')
df1.head(2)
df.set_index(df['sl_no'],inplace=True)
features=['gender','ssc_p','ssc_b','hsc_p','hsc_b','hsc_s','degree_p','degree_t','workex','etest_p','specialisation','status','mba_p']
sns.heatmap(df[features].corr(),linewidth=0.2,cmap="YlOrRd",annot=True)
sns.countplot(df['gender'],palette=['#FF7799','#AABBFF'])

sns.countplot(df["status"],palette=['#999900','#555555'])
plt.title("no of students placed")
plt.ylabel("no:of:students")
sns.catplot(x="status", y="ssc_p", jitter = False,data=df)
sns.catplot(x="status", y="hsc_p", jitter = False,data=df)
sns.catplot(x="status", y="degree_p", jitter = False,data=df)
sns.catplot(x="status", y="mba_p", jitter = False,data=df)

df1=df[df['status']==1]
sns.countplot(x="specialisation",data=df)
plt.title("specialisation")
plt.ylabel("no_of_students_place")
