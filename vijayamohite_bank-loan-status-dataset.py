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
credit_df=pd.read_csv("/kaggle/input/my-dataset/credit_train.csv")

credit_df.head()
credit_df.shape
credit_df.info()
credit_df.isnull().sum()
credit_df.describe()
credit_df['Number of Credit Problems'].plot()
credit_df.plot()
credit_df
credit_df.df = credit_df[credit_df['Credit Score']>800]

credit_df.head()
print("Value counts for each term: \n",credit_df['Term'].value_counts())

print("Missing data in loan term:",credit_df['Term'].isna().sum())
credit_df['Term'].replace(("Short Term","Long Term"),(0,1), inplace=True)

credit_df.head()
import pandas as pd
scount = credit_df[credit_df['Term'] == 0]['Term'].count()

lcount = credit_df[credit_df['Term'] ==1]['Term'].count()

data = {"Counts":[scount, lcount]}

credit_df = pd.DataFrame(data, index=["Short Term", "Long Term"])

credit_df.head()
credit_df.plot(kind="barh", title="Term of Loans")
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(font_scale = 2)

credit = pd.read_csv('/kaggle/input/my-dataset/credit_test.csv')
plt.figure(figsize=(20,8))



sns.countplot(credit['Years in current job'])
dataframe = pd.read_csv('/kaggle/input/my-dataset/credit_test.csv')

dataframe = dataframe.drop(['Credit Score'], axis=1)
dataframe['Purpose'].value_counts().sort_values(ascending=True).plot(kind='barh', title="Purpose for Loans", figsize=(15,10))