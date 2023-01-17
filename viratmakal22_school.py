# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/us-education-datasets-unification-project/states_all_extended.csv')

df.head()
label=df['YEAR'].unique()

size=df.groupby(['YEAR']).size()

print(size)

colors = ['#fae71b', '#66adff', 'darkslateblue']

plt.figure(figsize=(16,16))

plt.title("Percentage of Students in Year\n",color='red',fontsize=20)

diagram=plt.pie(size,colors=colors,labels=label,startangle=90,autopct='%1.1f%%')

plt.show()
Year1= df.groupby('YEAR').mean()

Year2= Year1['LOCAL_REVENUE']

colors1 = ['#fae71b', '#66adff', 'lightsteelblue']

plt.figure(figsize=(6,6))

plt.bar(df['YEAR'].unique(),Year1['LOCAL_REVENUE'], color = colors1)

plt.xticks(rotation=-45) 

plt.xlabel('\n\nYEAR') 

plt.ylabel('LOCAL_REVENUE') 

plt.title('Average\n') 

plt.show()



# No. of listings in each Year

flat = ['rosybrown','#fae71b','#66adff','maroon','lightsalmon']

sns.countplot(x='YEAR', data=df, palette=sns.color_palette(flat))

plt.title('No of Listings in each Year\n',fontsize=15,color='darkred') 

# Rotate x-labels

plt.xticks(rotation=45)

plt.xlabel('YEAR')

plt.ylabel('Listings')