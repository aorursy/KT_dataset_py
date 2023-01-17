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
import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 
df = pd.read_csv('/kaggle/input/health-insurance-cost-prediction/insurance.csv')

df.head()
df.info()
df['sex'].value_counts()
df['smoker'].value_counts()
df['region'].value_counts()
df['region'].value_counts().count()
region_count = df['region'].value_counts()

sns.barplot(region_count.index, region_count.values, alpha=0.9)

plt.title("Frequency distribution of region")

plt.ylabel("Frequency")

plt.xlabel("Region")

plt.show()
labels = df['region'].astype('category').cat.categories.tolist()

counts = df['region'].value_counts()

sizes = [counts[var_cat] for var_cat in labels]

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%')

ax1.axis('equal')

plt.show()
replace_map = {'region': {k:v for k,v in zip(labels,list(range(1,len(labels)+1)))}}

replace_map
df_replace = df.copy()

df_replace.replace(replace_map, inplace=True)

df_replace.head()
df_le = df.copy()

df_le['region'] = df_le['region'].astype('category')

df_le['region'] = df_le['region'].cat.codes

df_le.head()    # Alphabetically coded from 1 to 10

df_labelenc = df.copy()



from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

df_labelenc['region_code'] = le.fit_transform(df['region'])

df_labelenc.head()
df_onehot = df.copy()

df_onehot = pd.get_dummies(df_onehot, columns=['region'],prefix=['region'])

df_onehot.head()