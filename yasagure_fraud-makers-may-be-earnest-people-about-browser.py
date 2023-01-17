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
df.head()
pd.DataFrame(df.groupby("id_31").count()["TransactionID"]).head()
a = np.zeros(10)

a[:] = np.nan##this is numpy structure
a
a = np.zeros(df.shape[0])

a = np.nan

df["lastest_browser"] = a
df.head()
df.lastest_browser[df["id_31"]=="samsung browser 7.0"]=1

df.lastest_browser[df["id_31"]=="opera 53.0"]=1

df.lastest_browser[df["id_31"]=="mobile safari 10.0"]=1

df.lastest_browser[df["id_31"]=="google search application 49.0"]=1

df.lastest_browser[df["id_31"]=="firefox 60.0"]=1

df.lastest_browser[df["id_31"]=="edge 17.0"]=1

df.lastest_browser[df["id_31"]=="chrome 69.0"]=1

df.lastest_browser[df["id_31"]=="chrome 67.0 for android"]=1

df.lastest_browser[df["id_31"]=="chrome 63.0"]=1

df.lastest_browser[df["id_31"]=="chrome 63.0 for android"]=1

df.lastest_browser[df["id_31"]=="chrome 63.0 for ios"]=1

df.lastest_browser[df["id_31"]=="chrome 64.0"]=1

df.lastest_browser[df["id_31"]=="chrome 64.0 for android"]=1

df.lastest_browser[df["id_31"]=="chrome 64.0 for ios"]=1

df.lastest_browser[df["id_31"]=="chrome 65.0"]=1

df.lastest_browser[df["id_31"]=="chrome 65.0 for android"]=1

df.lastest_browser[df["id_31"]=="chrome 65.0 for ios"]=1

df.lastest_browser[df["id_31"]=="chrome 66.0"]=1

df.lastest_browser[df["id_31"]=="chrome 66.0 for android"]=1

df.lastest_browser[df["id_31"]=="chrome 66.0 for ios"]=1



df.lastest_browser[df.id_31=="samsung browser 7.0"]

df.lastest_browser[df["id_31"]=="chrome 66.0 for ios"]
df.isFraud.mean()
df.groupby("lastest_browser").mean()["isFraud"]
df["null"]=df.id_01.isnull()
df.groupby("null").mean()["isFraud"]