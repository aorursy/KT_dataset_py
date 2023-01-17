# Import Libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
df = pd.read_csv("../input/youtube-new/INvideos.csv")
df.head()
df.info()
# Finding correlation between different attributes



corrmat = df.corr()



plt.figure(figsize=(7,7))

sns.heatmap(df.corr(),annot=True,cmap='viridis')
corrmat['views'].sort_values(ascending=False)[1:]
plt.figure(figsize=(5,5))

sns.scatterplot('views','likes',data=df)
#Distribution of different continuous attributes

conti_features = [feature for feature in df.columns if df[feature].dtypes=='int64' and feature!='category_id']

for feature in conti_features:

    fig = sns.distplot(df[feature],hist=False,axlabel=feature)

    plt.show()