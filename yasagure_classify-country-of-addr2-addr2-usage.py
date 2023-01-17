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
pd.DataFrame(df.groupby("addr2").count()["TransactionID"])
import datetime

START_DATE = '2017-12-01'

startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')

df['TransactionDT'] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))

df['hour'] = df['TransactionDT'].dt.hour
df.groupby("addr2").mean()["hour"]
df.groupby("addr2").mean()["hour"] - 14.049229
def change(addr):

    if  addr==60.0:

        return 1

    elif addr==96.0:

        return 1

    elif addr == np.nan:

        return np.nan

    

    else:

        return 0



df["Europe"] = df["addr2"].map(change)
df.groupby("Europe").count().isFraud
586818*0.034585 + 3722*0.098872
df.groupby("Europe").mean().isFraud
def change(addr):

    if  addr==16.0:

        return 1

    elif addr==65.0:

        return 1

    elif addr == np.nan:

        return np.nan

    

    else:

        return 0



df["Asia"] = df["addr2"].map(change)
pd.DataFrame(df["Asia"]).info()
df.groupby("Asia").isFraud.sum()
df.groupby("Asia").isFraud.mean()
def change(addr):

    if  addr==31.0:

        return 1

    elif addr==32.0:

        return 1

    elif addr == np.nan:

        return np.nan

    

    else:

        return 0



df["North America"] = df["addr2"].map(change)
df.groupby("North America").isFraud.mean()
plt.ylim(0,0.4)

plt.subplot(2,2,1)

plt.bar([0,1],df.groupby("Asia").mean().isFraud.values)





plt.subplot(2,2,2)

plt.ylim(0,0.4)

plt.bar([0,1],df.groupby("Europe").mean().isFraud.values)





plt.subplot(2,2,3)

plt.ylim(0,0.4)

plt.bar([0,1],df.groupby("North America").mean().isFraud.values)