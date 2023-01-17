# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import missingno as mno

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

twice_company_data_filepath = "../input/twicecompanydata/Twice LTV Assignment Master.xlsx - Raw Data.csv"

twice_data = pd.read_csv(twice_company_data_filepath, index_col=0, encoding="latin-1")
#clean data: reset index, move column to last row, rename columns

twice_data = twice_data.reset_index()

columnToRow = twice_data.columns.tolist()

twice_data.rename(columns = {'Purchase':'transaction',"958507335":"id","2/1/2013":"date","2200":"amount"},inplace=True)

twice_data.loc["23158"] = columnToRow

twice_data.sample()
twice_data['amount'] = twice_data['amount'].astype(float)
purchase_data = twice_data[twice_data['transaction'] == 'Purchase']

payout_data = twice_data[twice_data['transaction'] == 'Sale']
purchase_data.columns
ltr = (purchase_data['amount'].sum()/purchase_data['id'].nunique())

ltp = (payout_data['amount'].sum()/payout_data['id'].nunique())
print("Lifetime Revenue: $" + str(round(ltr,2)))

print("Lifetime Payout: $" + str(round(ltp,2)))
#turn the user id column information from both dataframes into lists

seller_ids = payout_data['id'].unique().tolist()

buyer_ids = purchase_data['id'].unique().tolist()



#create new list to hold ids that buy and sell

superuser_ids = list()



#if a seller_id is also a buyer_id, add to list 

for each in seller_ids:

    if each in buyer_ids:

        superuser_ids.append(each)
#create a new dataframe with superuser ids

superuser_ids.sort()

superusers = pd.DataFrame({"id":superuser_ids})

#superusers.head()
superuser_buys = pd.merge(

    superusers,

    purchase_data,

    how='left',

    on='id'

)

superuser_buys = pd.DataFrame(superuser_buys.groupby(['id'])['amount'].sum())



superuser_sales = pd.merge(

    superusers,

    payout_data,

    how='left',

    on='id'

)

superuser_sales = pd.DataFrame(superuser_sales.groupby(['id'])['amount'].sum())
superusers = pd.merge(

    superuser_buys,

    superuser_sales,

    how='inner',

    on='id'

)

superusers.rename(columns={'amount_x':'purchases','amount_y':'sales'},inplace=True)

superusers.loc[:,'total'] = superusers['sales'] + superusers['purchases']
superusers.reset_index(inplace=True)

superuser_ltv = (superusers['total'].sum()/superusers['id'].count())
print("the ltv for users who both buy and sell is: $" + str(round(superuser_ltv,2)))
#for question 2, create table with the following columns:

#1 user id

#2 sum of all purchases

#3 value of first purchase

purchase_data.columns
#sort table so transactions occur in chronological order

first_transaction_data = purchase_data.sort_values('date')

#drop all subsequent orders from the same user id

first_transaction_data = first_transaction_data.drop_duplicates(subset ="id", keep='first')

first_transaction_data = first_transaction_data.filter(['id','amount'])

first_transaction_data = first_transaction_data.rename(columns={'amount':'first_transaction'})
user_transaction_volume = pd.DataFrame(purchase_data.groupby(['id'])['amount'].sum())

user_transaction_volume = user_transaction_volume.rename(columns={'amount':'total_transaction_volume'})
first_and_total_transaction_data = pd.merge(

    first_transaction_data,

    user_transaction_volume,

    on='id',

    how='inner'

)
#this boxplot reveals that removing some outliers is in order

g = sns.boxplot(x=first_and_total_transaction_data["first_transaction"], color='#e74c3c')
#remove outlier values. I had to play with these to get a visible hexplot.

quantile = first_and_total_transaction_data['first_transaction'].quantile(0.90)

first_and_total_transaction_data = first_and_total_transaction_data[first_and_total_transaction_data['first_transaction'] < quantile]

quantile = first_and_total_transaction_data['total_transaction_volume'].quantile(0.80)

first_and_total_transaction_data = first_and_total_transaction_data[first_and_total_transaction_data['total_transaction_volume'] < quantile]



g = sns.boxplot(x=first_and_total_transaction_data["first_transaction"], color='#e74c3c')
#this plot would imply that there is a weak correlation between first transaction and total transaction volume.

g = sns.jointplot(x='first_transaction', y='total_transaction_volume', 

                  data=first_and_total_transaction_data,kind='hex', color='#2ecc71')
#we can see that there is a strong positive correlation (over .80) 

first_and_total_transaction_data.corr()