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
data = pd.read_csv("../input/logistics-shopee-code-league/delivery_orders_march.csv")

SLA_data = pd.read_excel("../input/logistics-shopee-code-league/SLA_matrix.xlsx")
data[['1st_deliver_attempt','2nd_deliver_attempt','pick']] = data[['1st_deliver_attempt','2nd_deliver_attempt','pick']].astype('datetime64[s]')+pd.Timedelta('08:00:00')

data
#Remove unnecessary columns/rows

#SLA_data.drop(SLA_data.index[0], axis = 0, inplace=True)#This line is needed when using google colab

SLA_data.drop(SLA_data.columns[0], axis = 1, inplace=True)

SLA_data.dropna(how='all', inplace=True)



#Make 1st row into DataFrame header

SLA_data.columns = SLA_data.iloc[0]

#Rename seller origin column header (1st column) (NA --> 'origin')

SLA_data.rename(columns={SLA_data.columns[0]: 'origin'}, inplace = True)

#Drop 1st row

SLA_data.drop(SLA_data.index[0], inplace = True)



#Make 1st column into DataFrame index

Location_List =SLA_data['origin'].values.tolist()

SLA_data.index = Location_List

SLA_data.drop(columns=['origin'], inplace=True)



#Extract number

for i in SLA_data.columns:

  SLA_data[i] = SLA_data[i].str.extract('(\d+)').astype('int64')



#Unstack into series

SLA_data = SLA_data.stack()



#Convert multi-index into single index

new_idx = []



for i in range(0, len(SLA_data)):

    new_idx.append(" to ".join(SLA_data.index[i]))

    

SLA_data.index = new_idx



SLA_data
#List all location from SLA_data

patList = '('+'|'.join(Location_List)+')'



#Convert string to title case and extract location matched with the list

data.buyeraddress = data.buyeraddress.str.title()

data.buyeraddress = data.buyeraddress.str.findall(patList).str[-1]



data.selleraddress = data.selleraddress.str.title()

data.selleraddress = data.selleraddress.str.findall(patList).str[-1]
data
#define custom holidays

holid = ['2020-03-08', '2020-03-25', '2020-03-30', '2020-03-31']

for i in range(0, len(holid)):

  holid[i] = np.datetime64(holid[i], 'D')
#Add new column and calculate the value

data['1st_interval'] = np.busday_count(begindates=data['pick'].values.astype('datetime64[D]'),

                    enddates=data['1st_deliver_attempt'].values.astype('datetime64[D]'), 

                    weekmask='Mon Tue Wed Thu Fri Sat', holidays=holid)
#Add new column with initiate value -1

data['2nd_interval'] = -1



#Get boolean index of rows with 2nd attempt available

idx_2nd_attempt = pd.notna(data['2nd_deliver_attempt'])





data.loc[idx_2nd_attempt,'2nd_interval'] = np.busday_count(begindates=data.loc[idx_2nd_attempt,'1st_deliver_attempt'].values.astype('datetime64[D]'),

                    enddates=data.loc[idx_2nd_attempt,'2nd_deliver_attempt'].values.astype('datetime64[D]'), 

                    weekmask='Mon Tue Wed Thu Fri Sat', holidays=holid)
data
data['SLA'] = data['selleraddress'] + " to " + data['buyeraddress']

data['SLA'] = data['SLA'].map(SLA_data)
#Evaluate first attempt

data['is_late'] = data['1st_interval'] > data['SLA']



#Extract order failed 1st attempt but isn't late on that attempt

idx_failed_1st = (data['is_late'] == 0) & (idx_2nd_attempt)



#Evaluate 2nd attempt

data.loc[idx_failed_1st, 'is_late'] = data.loc[idx_failed_1st,'2nd_interval'] > 3



#Convert boolean to int

data['is_late'] = data['is_late'].astype('int64')



data
data[['orderid', 'is_late']].to_csv('submission.csv', header = True, index = False)