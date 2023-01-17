# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
app=pd.read_csv("../input/app-store-apple-data-set-10k-apps/AppleStore.csv")
app.head()
app.info
app.isnull().values.any()
app.loc[lambda df: df["price"] == 0]
free_app=app.loc[lambda df: df["price"] == 0]
app.loc[lambda df: df["price"] >0]
paid_app=app.loc[lambda df: df["price"] > 0]
sns.catplot(y="user_rating",kind="violin",data=paid_app);
plt.figure(figsize=(15,5))
sns.boxplot(y="lang.num",x="ipadSc_urls.num",data=paid_app);
plt.figure(figsize=(14,7))
sns.scatterplot(y="lang.num",x="ipadSc_urls.num",data=paid_app);
plt.figure(figsize=(15,4))
categories=app["prime_genre"].value_counts()
sns.barplot(x=categories[:10].index,y=categories[:10].values);

plt.figure(figsize=(15,4))
categories_paid_app=paid_app["prime_genre"].value_counts()
sns.barplot(x=categories_paid_app[:10].index,y=categories_paid_app[:10].values);

plt.figure(figsize=(15,4))
categories_free_app=free_app["prime_genre"].value_counts()
sns.barplot(x=categories_free_app[:10].index,y=categories_free_app[:10].values);
