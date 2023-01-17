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
train=pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip", parse_dates=["Date"])

test=pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip", parse_dates=["Date"])

stores=pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv")

features=pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip")
train.head(2)
stores.head(2)
features.head(2)
train['Date'] =pd.to_datetime(train['Date'], format="%Y-%m-%d")

features['Date'] =pd.to_datetime(features['Date'], format="%Y-%m-%d")

test['Date'] = pd.to_datetime(test['Date'], format="%Y-%m-%d")
combined_train = pd.merge(train,stores,how='left',on='Store')

combined_train.head(2)
combined_test = pd.merge(test,stores,how='left',on='Store')

combined_test.head(2)
combined_train = pd.merge(combined_train, features,how = "inner", on=["Store","Date",'IsHoliday'])

combined_train.head(2)
combined_train.shape
combined_test = pd.merge(combined_test, features, how = "inner", on=["Store","Date",'IsHoliday'])

combined_test.head(2)
combined_train.isna().sum()
combined_train.fillna(0,inplace=True)
combined_train.describe()
combined_train.info()
import seaborn as sns

import matplotlib.pyplot as plt

f, axes = plt.subplots(1, 2)

sns.kdeplot(combined_train['Weekly_Sales'], ax=axes[0])

sns.boxplot(combined_train['Weekly_Sales'], ax=axes[1])

plt.show()
combined_train['Date'] = combined_train['Date'].astype('datetime64[ns]').astype(int)
ax = sns.boxplot(data = combined_train, orient = "h", color = "violet", palette = "Set1")

plt.show()
combined_train.head(2)