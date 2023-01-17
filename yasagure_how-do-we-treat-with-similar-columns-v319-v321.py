import pandas as pd

import numpy as np

import matplotlib.pylab as plt

%matplotlib inline

pd.set_option('display.max_rows', 500)

pd.get_option("display.max_columns",500)
folder_path = '../input/'

train_identity = pd.read_csv(f'{folder_path}train_identity.csv')

train_transaction = pd.read_csv(f'{folder_path}train_transaction.csv')





df = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
df.corr().head()
cor = df.corr()
cor_use = cor[cor>0.8]
for i in range(cor_use.shape[0]):

    cor_use.iloc[i,i] = np.nan
cor_use.head()
corr_use = cor_use[cor_use>0.7].dropna(how='all').dropna(how='all',axis=1)
corr_use.shape
df.loc[:,["V319","V320","V321"]].corr()
df.loc[:,['V319','V320','V321']].head(30)
df["diff_V319_V320"] = np.zeros(df.shape[0])

df["diff_V320_V321"] = np.zeros(df.shape[0])

df["diff_V319_V321"] = np.zeros(df.shape[0])
len(df[(df["V319"]!=df["V320"])])/df.shape[0]
df.loc[df["V319"]!=df["V320"],"diff_V319_V320"] = 1
df[(df["V319"]!=df["V320"])].head()
df.groupby("diff_V319_V320").mean().isFraud
df.groupby("diff_V319_V320").mean().isFraud.plot()
len(df[(df["V320"]!=df["V321"])])/df.shape[0]
df.loc[df["V321"]!=df["V320"],"diff_V320_V321"] = 1
df[(df["V321"]!=df["V320"])].head()
df.groupby("diff_V320_V321").mean().isFraud
len(df[(df["V319"]!=df["V321"])])/df.shape[0]
df.loc[df["V321"]!=df["V319"],"diff_V319_V321"] = 1
df[(df["V321"]!=df["V320"])].head()
df.groupby("diff_V320_V321").mean().isFraud