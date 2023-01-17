# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
print("Reading csv's...")    

train = pd.read_csv('../input/train_transaction.csv')

test = pd.read_csv('../input/test_transaction.csv')

traini = pd.read_csv('../input/train_identity.csv') 

testi = pd.read_csv('../input/test_identity.csv')

print('Done!')
train = pd.merge(train, traini, on='TransactionID', how='left')

test = pd.merge(test, testi, on='TransactionID', how='left')

del traini

del testi
def make_hour_feature(df, tname='TransactionDT'):

    """

    Creates an hour of the day feature, encoded as 0-23. 

    

    Parameters:

    -----------

    df : pd.DataFrame

        df to manipulate.

    tname : str

        Name of the time column in df.

    """

    hours = df[tname] / (3600)        

    encoded_hours = np.floor(hours) % 24

    return encoded_hours



train['hours'] = make_hour_feature(train)

test['hours'] = make_hour_feature(test)



print("Added hours feature...")
from sklearn.utils import resample



print("Fraud counts",len(train[train.isFraud == 1]))

print("Non Fraud counts",len(train[train.isFraud == 0]))



for h in range(24):

    fc = len(train[(train.isFraud==1) &(train.hours==h)])

    nfc = len(train[(train.isFraud==0) &(train.hours==h)])

    print("hour ",h,"fraud counts:",fc)

    print("hour ",h,"non-fraud counts:",nfc)

    if fc < nfc and fc > 0: # on small dataset we may need this check

        print("Fraud precentage ceil:",int(np.ceil(100*fc/nfc)))    

        chunk = train[train.hours==h]

        df_majority = chunk[chunk.isFraud==0]

        df_minority = chunk[chunk.isFraud==1]

        maj_len = len(df_majority) 

        min_len = len(df_minority) 

        # Upsample minority class

        df_minority_upsampled = resample(df_minority, 

                                     replace=True,     # sample with replacement

                                     n_samples=maj_len,    # to match majority class

                                     random_state=123) # reproducible results

     

    # Combine majority class with upsampled minority class

        train = pd.concat([train, df_minority_upsampled])

        del df_majority

        del df_minority

        del df_minority_upsampled

print("Fraud counts",len(train[train.isFraud == 1]))

print("Non Fraud counts",len(train[train.isFraud == 0]))

train = train.sample(frac=1,random_state=100)  # reset index      

train = train.sort_values('hours')

test = test.sort_values('hours')