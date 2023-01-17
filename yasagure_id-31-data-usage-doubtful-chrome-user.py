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
pd.DataFrame(df.groupby("id_31").count().TransactionID)
pd.DataFrame(df.groupby("id_31").count().TransactionID).plot()
df["chrome"]=df['id_31'].str.contains('chrome')*1

##here, I cange bool data to numeric data by mutipling 1( * 1)
df["chrome"].head()
df["samsung_browser"] = df['id_31'].str.contains('samsung')*1
df["safari"] = df['id_31'].str.contains('safari')*1
df["opera"] = df['id_31'].str.contains('opera')*1
df["ie"] = df['id_31'].str.contains('ie')*1
df["google_browser"] = df['id_31'].str.contains('google')*1
df["firefox"] = df['id_31'].str.contains('firefox')*1
df["edge"] = df['id_31'].str.contains('edge')*1
df["android_browser"] = df['id_31'].str.contains('android browser')*1

df["android_browser"] = df['id_31'].str.contains('android webview')*1

df["android_browser"] = df['id_31'].str.contains('Generic/Android')*1

df["android_browser"] = df['id_31'].str.contains('Generic/Android 7.0')*1
df.groupby("chrome").mean().isFraud
df.groupby("safari").mean().isFraud
pd.get_option("display.max_columns",500)

pd.options.display.max_columns = None
df.head()
df.groupby("DeviceType").mean().isFraud
df.groupby("edge").mean().isFraud
df.isFraud.mean()
df.groupby("isFraud").count().TransactionID
df.shape
pd.DataFrame(df["id_31"]).info( all,null_counts=True)