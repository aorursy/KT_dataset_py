# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

# packages

import matplotlib.pyplot as plt

#import mpld3

import warnings

import seaborn as sns

sns.set(style='darkgrid')



#

warnings.filterwarnings('ignore')

#mpld3.enable_notebook()

#

# sales data, lets load the data

train = pd.read_csv('../input/train.csv')

# sort the dates 

train.sort_values(by='Date', ascending=True, inplace=True)

# stores data

stores = pd.read_csv('../input/store.csv')

#

print('-----Train data ------------------------------------')

print(train.head(10))

print('-----------------------------------------')

print('-----Stores data ------------------------------------')

print(stores.head(10))

print('-----------------------------------------')

#

# lets collate sales and customer data on monthly and yearly charts

# split Year-Month-Date to three different columns

train['Month'] = train['Date'].apply(lambda x : int(str(x)[5:7]))

train['Year'] = train['Date'].apply(lambda x : int(str(x)[:4]))

train['MonthYear'] = train['Date'].apply(lambda x : (str(x)[:7]))



#

train.info()

stores.info()
train['LogSales'] = np.log(train['Sales']+1)  # +1 to take care of log(0) condition

train_stores = train[train['Open']!=0]

cols = ['DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 'Month','Year']  # interested in these attributes



fig, axis = plt.subplots(3,2, figsize=(15,15))

axis = axis.ravel()

for i, attr in enumerate(cols):

    ax1 = sns.boxplot(x=attr, y='Sales', data=train_stores[['Sales', attr]], palette='husl', ax=axis[i])

    axis[i].set_title('Distributions per ' + attr)
import datetime

# stores data

stores = pd.read_csv('../input/store.csv')

# there are many NANs, remove them

stores_notnull = stores['CompetitionOpenSinceMonth'].notnull() & stores['CompetitionOpenSinceYear'].notnull()

# create CompetitionOpenDate

stores['CompOpen'] = stores[stores_notnull]['CompetitionOpenSinceYear'].astype(int).astype(str).str.cat(stores[stores_notnull]['CompetitionOpenSinceMonth'].astype(int).astype(str).str.zfill(2), sep='-')

stores['CompOpenDate'] = pd.Series([datetime.datetime.strptime(str(ym), '%Y-%m').strftime('%Y-%m-%d') for ym in stores[stores_notnull]['CompOpen'].tolist()], index = stores[stores_notnull].index)    

# fill CompetitionDistance for Nan as high number

stores['CompetitionDistance'].fillna(value=1000000, inplace=True)





# let's update train data

# create a step function based on CompOpenDate for train

print('processing Stores...')

for store in stores['Store']:

    print('\r', 'Store: ', store, end='')

    storedata = train[train['Store'] == store]

    compd = stores[stores['Store']==store]['CompOpenDate']

    dist = stores[stores['Store']==store]['CompetitionDistance']

    train.ix[train['Store']==store, 'CompImpact'] = (storedata['Date'] > compd.values[0]).astype(int).values * (1/dist.values[0])



print('\n','finished')
# there are many NANs, remove them

stores_notnull = stores['Promo2SinceWeek'].notnull() & stores['Promo2SinceYear'].notnull()

# create Promo2OpenDate

stores['Promo2Open'] = stores[stores_notnull]['Promo2SinceYear'].astype(int).astype(str).str.cat(stores[stores_notnull]['Promo2SinceWeek'].astype(int).astype(str).str.zfill(2), sep='-')

stores['Promo2OpenDate'] = pd.Series([datetime.datetime.strptime(str(ym)+'-0', '%Y-%W-%w').strftime('%Y-%m-%d') for ym in stores[stores_notnull]['Promo2Open'].tolist()], index = stores[stores_notnull].index)    



# let's update train data

# create a step function based on CompOpenDate for train

print('processing Stores...')

for store in stores['Store']:

    print('\r', 'Store: ', store, end='')

    storedata = train[train['Store'] == store]

    p2d = stores[stores['Store']==store]['Promo2OpenDate']

    train.ix[train['Store']==store, 'Promo2'] = (storedata['Date'] > p2d.values[0]).astype(int).values





print('\n','finished')
# plot

fig, (axis1, axis2, axis3) = plt.subplots(3,1, sharex=True, figsize=(10,7))

# We will now plot the generated series for one store to see if it looks alright.

# Pick a random store which has CompOpenDate and Promo2OpenDate  

store = stores[(stores['CompOpenDate'].notnull()) & (stores['Promo2OpenDate'].notnull())]['Store'].sample(n=1).values[0]



# display CompOpenDate and Promo2OpenDate

print('Competition Open Date: ' + stores[stores['Store']==store]['CompOpenDate'].astype(str))

print('Promo2 Start Date: ' + stores[stores['Store']==store]['Promo2OpenDate'].astype(str))

# plot generated series along with sales

storedata = train[train['Store']==store]

storedata['Date'] = pd.to_datetime(storedata['Date'])

#

storedata['CompImpact'].plot(marker='o', ax=axis1)

tmp = axis1.set_title('Store-{} :Competition Impact'.format(store))

storedata['Promo2'].plot(marker='o', ax=axis2)

tmp = axis2.set_title('Promo2 Start')

storedata['Sales'].plot(marker='o', ax=axis3)

tmp = axis3.set_title('Sales')

tmp = axis3.set_xticks(storedata['Date'].index[::20])

tmp = axis3.set_xticklabels(storedata['Date'][::20], rotation=90)
# StoreType and Assortment

print('processing Stores...')

for store in stores['Store']:

    print('\r', 'Store: ', store, end='')

    storedata = train[train['Store'] == store]

    st = stores[stores['Store']==store]['StoreType']

    train.ix[train['Store']==store, 'StoreType'] = st.values[0]

    asst = stores[stores['Store']==store]['Assortment']

    train.ix[train['Store']==store, 'Assortment'] = asst.values[0]

    

print('\n','finished')
## lets prepare data for training and testing



# make a copy of the data 

train_copy = train.copy()



# StateHoliday, StoreType and Assortment are categorical strings, convert using simple LabelEncoder/DictVectoriser

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction import DictVectorizer

train_copy['StateHoliday'].replace(0, '0', inplace=True)

train_copy['StateHoliday'] = LabelEncoder().fit_transform(train_copy['StateHoliday'])

train_copy['StoreType'] = LabelEncoder().fit_transform(train_copy['StoreType'])

train_copy['Assortment'] = LabelEncoder().fit_transform(train_copy['Assortment'])



# Date is already sorted

unique_dates = train_copy['Date'].unique()

train_length = np.round(unique_dates.shape[0] * 0.8).astype(int)

#

train_data = train_copy[train_copy['Date'].isin(unique_dates[0:train_length])]

test_data = train_copy[train_copy['Date'].isin(unique_dates[train_length+1:])]

# feature attributes

feature_attributes = ['Open', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 'CompImpact', 'Promo2', 'Month', 'Year', 'StoreType', 'Assortment']

target_attribute = ['LogSales']

#
