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

data=pd.read_csv('../input/world-foodfeed-production/FAO.csv',encoding = "ISO-8859-1")
data.head()
data.shape
data.describe()
print(data.isnull().sum())

data['Element Code'].plot(kind='hist',bins=30)
data.iloc[0:4,0:11]
area=data['Area'].unique()
anually=data.iloc[:,10:]


for i in area:

    list1=[]

    for j in anually:

        list1.append(data[j][data['Area']==i].sum())

        plt.plot(list1,label=i)
sns.factorplot('Element',data=data,kind='count')
sns.factorplot('Area',data=data[(data['Area']=='India')|(data['Area']=='United States of America')],kind='count',hue='Element')
new_df_dict={}

for i in area:

    list1=[]

    for j in anually:

        list1.append(data[j][data['Area']==i].sum())

        new_df_dict[i] = list1

new_df = pd.DataFrame(new_df_dict)





new_df.head()
new_df = pd.DataFrame.transpose(new_df)