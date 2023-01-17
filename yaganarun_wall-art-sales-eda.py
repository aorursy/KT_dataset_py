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
import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import re
df = pd.read_csv('../input/wall-art-sales/Wall Art sales - Sheet1.csv' , engine = 'c')
df.head()
df.info()
# Since Discount has 10 NULL values it must be dropped

print(df.isnull().sum())

df = df.dropna()
duplicate_rows = df.duplicated()

df[ duplicate_rows ]
# Droping redundant rows 

df = df.drop_duplicates()
# renaming the columns 

df = df.rename(columns = {'Link':'Product' , 'Shipping' : 'Shipping_Days'})
# Modifying the dataset for EDA

df['Shipping_Days'] = df['Shipping_Days'].apply( lambda x : int(re.findall( '\d+' , x )[0]) )

df['Discount'] = df['Discount'].apply(lambda x : int(re.findall('\d+' , x )[0]))

df['Price'] = df['Price'].apply(lambda x : int(re.sub(',','',x)[1:]))

df['Brand'] = df['Brand'].apply( lambda x: re.sub('by' , '' , x))

df.head()
# Identifying how strong each variable are related

sns.heatmap(df.corr())
# Maximum discount provided by each Brand

px.bar(df.groupby(['Brand']).max()['Discount'])
# Minimum discount provided by each brand

px.bar(df.groupby(['Brand']).min()['Discount'])
# Maximum discount for craft item with different prices

px.scatter(df.groupby(['Price']).max()['Discount'] , labels = {'value':'Discount'})