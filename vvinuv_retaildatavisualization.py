# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# visualization

import seaborn as sns

import matplotlib.pyplot as pl

import calendar   
feature_file = '../input/Features data set.csv'

sales_file = '../input/sales data-set.csv'

stores_file = '../input/stores data-set.csv'

feature_data = pd.read_csv(feature_file)

sales_data = pd.read_csv(sales_file)

stores_data = pd.read_csv(stores_file)
print(feature_data.columns.tolist())
print(sales_data.columns.tolist())
print(stores_data.columns.tolist())
sales_data['Date'] = pd.to_datetime(sales_data['Date'])

feature_data['Date'] = pd.to_datetime(feature_data['Date'])
fig, axarr = pl.subplots(7, 7, sharex=True, sharey=True,figsize=(15,10))

s = 1

for i in range(0, 7):

    for j in range(0, 7):

        xxx = axarr[i,j].hist(sales_data['Weekly_Sales'].loc[sales_data['Store'] == s], 50);

        axarr[i,j].set_yscale('log')

        axarr[i,j].set_xscale('log')

        axarr[i,j].set_ylim(1,1e4)

        axarr[i,j].set_xlim(5e2,1e6)



        s += 1

fig.text(0.5, 0.04, 'Weekly Sales', ha='center')

fig.text(0.04, 0.5, 'Number', va='center', rotation='vertical')
fig, axarr = pl.subplots(7, 7, sharex=True, sharey=True,figsize=(15,10))

s = 1

for i in range(0, 7):

    for j in range(0, 7):

        xxx = axarr[i,j].hist(sales_data['Weekly_Sales'].loc[(sales_data['Store'] == s) & 

                             (sales_data['IsHoliday'] == False)], 20, color='b', normed=True, histtype='step')

        

        xxx = axarr[i,j].hist(sales_data['Weekly_Sales'].loc[(sales_data['Store'] == s) & 

                             (sales_data['IsHoliday'] == True)], 20, color='r', normed=True, histtype='step')

        axarr[i,j].set_yscale('log')

        axarr[i,j].set_xscale('log')

        #axarr[i,j].set_ylim(1,1e4)

        axarr[i,j].set_ylim(1e-10,5e-4)

        axarr[i,j].set_xlim(5e2,1e6)



        s += 1

        

fig.text(0.5, 0.04, 'Weekly Sales', ha='center')

fig.text(0.04, 0.5, 'Normalized umber', va='center', rotation='vertical')
fig, axarr = pl.subplots(4, 3, sharex=True, sharey=True,figsize=(15,10))

s, m = 1, 1

for i in range(0, 4):

    for j in range(0, 3):

        xxx = axarr[i,j].hist(sales_data['Weekly_Sales'].loc[(sales_data['Store'] == s) & 

                            (sales_data['Date'].dt.month == m)], 20, normed=True);

        axarr[i,j].set_yscale('log')

        axarr[i,j].set_xscale('log')

        axarr[i,j].set_xlim(5e2,1e6)

        #axarr[i,j].set_ylim(1,1e4)

        axarr[i,j].set_ylim(1e-10,1e-4)

        axarr[i,j].set_title('%s'%calendar.month_name[m])

        m += 1

fig.text(0.5, 0.04, 'Weekly Sales', ha='center')

fig.text(0.04, 0.5, 'Number', va='center', rotation='vertical')
fig, axarr = pl.subplots(4, 3, sharex=True, sharey=True,figsize=(15,10))

s, m = 1, 1

for i in range(0, 4):

    for j in range(0, 3):

        xxx = axarr[i,j].hist(sales_data['Weekly_Sales'].loc[(sales_data['Store'] == s) & 

                             (sales_data['Date'].dt.year == 2010) & 

                             (sales_data['Date'].dt.month == m)], 20, color='b', 

                              histtype='step', normed=True);

        xxx = axarr[i,j].hist(sales_data['Weekly_Sales'].loc[(sales_data['Store'] == s) & 

                             (sales_data['Date'].dt.year == 2011) & 

                             (sales_data['Date'].dt.month == m)], 20, color='g', 

                              histtype='step', normed=True);

        xxx = axarr[i,j].hist(sales_data['Weekly_Sales'].loc[(sales_data['Store'] == s) & 

                             (sales_data['Date'].dt.year == 2012) & 

                             (sales_data['Date'].dt.month == m)], 20, color='r', 

                              histtype='step', normed=True);



        axarr[i,j].set_yscale('log')

        axarr[i,j].set_xscale('log')

        axarr[i,j].set_ylim(1e-10,1e-4)

        axarr[i,j].set_xlim(5e2,1e6)

        axarr[i,j].set_title('%s'%calendar.month_name[m])

        m += 1

fig.text(0.5, 0.04, 'Weekly Sales', ha='center')

fig.text(0.04, 0.5, 'Number', va='center', rotation='vertical')
stores_data['SizeBand'] = pd.cut(stores_data['Size'], bins=4, labels=np.arange(1, 5)).astype(np.int)
StoreSizeDict = stores_data.set_index('Store').to_dict()['SizeBand']

StoreTypeDict = stores_data.set_index('Store').to_dict()['Type']
sales_data['SizeBand'] = sales_data['Store']

sales_data['SizeBand'] = sales_data['SizeBand'].map(StoreSizeDict)

sales_data['Type'] = sales_data['Store'].map(StoreTypeDict)
fig, axarr = pl.subplots(4, 3, sharex=True, sharey=True,figsize=(15,10))

s, m = 1, 1

for i in range(0, 4):

    for j in range(0, 3):

        tdf = sales_data.loc[(sales_data['Store'] == s) & (sales_data['Date'].dt.month == m)]

        tdf = tdf.groupby('Date')['Weekly_Sales'].sum().reset_index()

        xxx = axarr[i,j].hist(tdf['Weekly_Sales'], 20);

        axarr[i,j].set_title('%s'%calendar.month_name[m])

        axarr[i,j].set_xscale('log')

        axarr[i,j].set_xlim(5e5,5e6)

        m += 1

fig.text(0.5, 0.04, 'Sum of weekly Sales', ha='center')

fig.text(0.04, 0.5, 'Number', va='center', rotation='vertical')
fig, axarr = pl.subplots(4, 3, sharex=True, sharey=True,figsize=(15,10))

s, m = 1, 1

for i in range(0, 4):

    for j in range(0, 3):

        tdf = sales_data.loc[(sales_data['Store'] == s) 

                             & (sales_data['Date'].dt.month == m) 

                             & (sales_data['Date'].dt.year == 2010)]

        tdf = tdf.groupby('Date')['Weekly_Sales'].sum().reset_index()

        xxx = axarr[i,j].hist(tdf['Weekly_Sales'], 20, color='b');#, histtype='step');

        

        tdf = sales_data.loc[(sales_data['Store'] == s) 

                             & (sales_data['Date'].dt.month == m) 

                             & (sales_data['Date'].dt.year == 2011)]

        tdf = tdf.groupby('Date')['Weekly_Sales'].sum().reset_index()

        xxx = axarr[i,j].hist(tdf['Weekly_Sales'], 20, color='g');#, histtype='step');

        

        tdf = sales_data.loc[(sales_data['Store'] == s) 

                             & (sales_data['Date'].dt.month == m) 

                             & (sales_data['Date'].dt.year == 2012)]

        tdf = tdf.groupby('Date')['Weekly_Sales'].sum().reset_index()

        xxx = axarr[i,j].hist(tdf['Weekly_Sales'], 20, color='r');#, histtype='step');

        axarr[i,j].set_title('%s'%calendar.month_name[m])

        axarr[i,j].set_xscale('log')

        axarr[i,j].set_xlim(5e5,5e6)

        m += 1

fig.text(0.5, 0.04, 'Sum of weekly Sales', ha='center')

fig.text(0.04, 0.5, 'Number', va='center', rotation='vertical')
fig, axarr = pl.subplots(2, 2, sharex=True, sharey=True,figsize=(15,10))

s, m = 1, 1

for i in range(0, 2):

    for j in range(0, 2):

        xxx = axarr[i,j].hist(sales_data['Weekly_Sales'].loc[(sales_data['SizeBand'] == m)], 

                              20, normed=True)

        axarr[i,j].set_yscale('log')

        axarr[i,j].set_xscale('log')

        #axarr[i,j].set_ylim(1,1e6)

        axarr[i,j].set_xlim(5e2,1e6)

        axarr[i,j].set_title('Size Band = %d'%m)

        m += 1
fig, axarr = pl.subplots(2, 2, sharex=True, sharey=True,figsize=(15,10))

s, m = 1, 1

for i in range(0, 2):

    for j in range(0, 2):

        xxx = axarr[i,j].hist(sales_data['Weekly_Sales'].loc[(sales_data['SizeBand'] == m) & 

                             (sales_data['IsHoliday'] == False)], 50, color='b', 

                              normed=True, histtype='step')

        xxx = axarr[i,j].hist(sales_data['Weekly_Sales'].loc[(sales_data['SizeBand'] == m) & 

                             (sales_data['IsHoliday'] == True)], 50, color='r', 

                              normed=True, histtype='step')

        axarr[i,j].set_yscale('log')

        axarr[i,j].set_xscale('log')

        #axarr[i,j].set_ylim(1,1e4)

        axarr[i,j].set_xlim(5e2,1e6)



        m += 1

fig.text(0.5, 0.04, 'Weekly Sales', ha='center')

fig.text(0.04, 0.5, 'Normalized umber (Red/Blue:Holiday/Non)', va='center', rotation='vertical')
fig, axarr = pl.subplots(4, 3, sharex=True, sharey=True,figsize=(15,10))

s, m = ['A', 'B', 'C'], 1

for i in range(0, 4):

    for j in range(0, 3):

        xxx = axarr[i,j].hist(sales_data['Weekly_Sales'].loc[(sales_data['SizeBand'] == m) 

                              & (sales_data['Type'] == s[j]) & (sales_data['IsHoliday'] == False)], 50, 

                              normed=True, color='b', histtype='step');

        xxx = axarr[i,j].hist(sales_data['Weekly_Sales'].loc[(sales_data['SizeBand'] == m) 

                              & (sales_data['Type'] == s[j]) & (sales_data['IsHoliday'] == True)], 50, 

                              normed=True, color='r', histtype='step');

        axarr[i,j].set_yscale('log')

        axarr[i,j].set_xscale('log')

        axarr[i,j].set_ylim(1e-10,1e-4)

        axarr[i,j].set_xlim(5e2,1e6)

        axarr[i,j].set_title('SizeBand=%d, Type=%s'%(m, s[j]))

    m += 1

fig.text(0.5, 0.04, 'Weekly Sales', ha='center')

fig.text(0.04, 0.5, 'Normalized umber (Red/Blue:Holiday/Non)', va='center', rotation='vertical')
sales_feature = sales_data.merge(feature_data, left_on=('Store', 'Date'), 

                                 right_on=('Store', 'Date'), how='left')
sales_feature.columns.tolist()
sales_feature.Type.unique()
sales_feature = sales_feature.drop(['IsHoliday_y'], axis=1)

sales_feature = sales_feature.rename(columns = {'IsHoliday_x':'IsHoliday'})

sales_feature['IsHoliday'] = sales_feature['IsHoliday'].astype(int)
# Compute the correlation matrix

corr = sales_feature.drop(['Store', 'Dept'], axis=1).corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True
#correlation matrix

f, ax = pl.subplots(figsize=(12, 9))

sns.heatmap(sales_feature.drop(['Store', 'Dept'], axis=1).corr(), mask=mask, vmax=.8, square=True);
#correlation matrix

fig, axarr = pl.subplots(1,3, figsize=(15, 3.5))

sns.heatmap(sales_feature.drop(['Store', 'Dept'], axis=1)[sales_feature.Type == 'A'].corr(), mask=mask, vmax=.6, 

            square=True, ax=axarr[0], cbar=None);

sns.heatmap(sales_feature.drop(['Store', 'Dept'], axis=1)[sales_feature.Type == 'B'].corr(), mask=mask, vmax=.6, 

            square=True, ax=axarr[1], cbar=None);

sns.heatmap(sales_feature.drop(['Store', 'Dept'], axis=1)[sales_feature.Type == 'C'].corr(), mask=mask, vmax=.6, 

            square=True, ax=axarr[2], cbar=None);