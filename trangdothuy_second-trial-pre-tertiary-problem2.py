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
orders = pd.read_csv('/kaggle/input/ptr-rd2-ahy/orders.csv', low_memory= False)

devices = pd.read_csv('/kaggle/input/ptr-rd2-ahy/devices.csv',low_memory=False)

bank_accounts = pd.read_csv('/kaggle/input/ptr-rd2-ahy/bank_accounts.csv',low_memory= False)

credit_cards = pd.read_csv('/kaggle/input/ptr-rd2-ahy/credit_cards.csv',low_memory= False)
from tqdm.notebook import tqdm
final_df=devices.merge(bank_accounts,how='left',on ='userid').merge(credit_cards,how='left',on='userid')

final_df.head()
orders['buyers_seller']=list(zip(orders['buyer_userid'],orders['seller_userid']))

orders.head()
def fraud_detection(userid_a, userid_b):

    storage_a=[]

    for device in devices[devices['userid']==userid_a].device:

        if device not in storage_a:

            storage_a.append(device)

            

    for credit_card in credit_cards[credit_cards['userid']==userid_a].credit_card:

        if (credit_card not in storage_a)&(pd.notnull(credit_card)==True):

            storage_a.append(credit_card)

            

    for bank_account in bank_accounts[bank_accounts['userid']==userid_a].bank_account:

        if (bank_account not in storage_a)&(pd.notnull(bank_account)==True):

            storage_a.append(bank_account)

    

    storage_b=[]

    for device in devices[devices['userid']==userid_b].device:

        if device not in storage_b:

            storage_b.append(device)

            

    for credit_card in credit_cards[credit_cards['userid']==userid_b].credit_card:

        if (credit_card not in storage_b)&(pd.notnull(credit_card)==True):

            storage_b.append(credit_card)

            

    for bank_account in bank_accounts[bank_accounts['userid']==userid_b].bank_account:

        if (bank_account not in storage_b)&(pd.notnull(bank_account)==True):

            storage_b.append(bank_account)

            

    storage_a_series = pd.Series(storage_a)

    storage_b_series = pd.Series(storage_b)

    result = storage_a_series.isin(storage_b_series).sum()

    if result>0:

        return 1

    else:

        return 0

            
fraud_detection(26855196, 16416890)
is_fraud=[]

for buyer_userid, seller_userid in order['buyer_seller']:

    result = fraud_detection(buyer_userid, seller_userid)

    is_fraud.append(result)

orders['is_fraud']=is_fraud

orders[['orderid','is_fraud']].to_csv('submission.csv', index=False)