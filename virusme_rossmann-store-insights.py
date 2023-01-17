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

import numpy as np

import pandas as pd

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

train.head(10)
ax = sns.countplot(x='Open', hue='DayOfWeek', data=train, palette='Set1')
# lets collate sales and customer data on monthly and yearly charts

# split Year-Month-Date to three different columns

train['Month'] = train['Date'].apply(lambda x : int(str(x)[5:7]))

train['Year'] = train['Date'].apply(lambda x : int(str(x)[:4]))

train['MonthYear'] = train['Date'].apply(lambda x : (str(x)[:7]))

                                    

# median sales

median_sales = train.groupby('MonthYear')['Sales'].median()

pct_median_change = train.groupby('MonthYear')['Sales'].median().pct_change()

# median customers

median_cust = train.groupby('MonthYear')['Customers'].median()

pct_median_custchange = train.groupby('MonthYear')['Customers'].median().pct_change()



fig, (axis1, axis2) = plt.subplots(2, 1, sharex=True, figsize=(10,5))

# plot median sales

ax1 = median_sales.plot(legend=True, ax=axis1, marker='o',title="Median")

ax1.set_xticks(range(len(median_sales)))

ax1.set_xticklabels(median_sales.index.tolist(), rotation=90)

#pd.rolling_mean(median_sales, window=7).plot(ax=axis1)

# plot pct change

ax2 = pct_median_change.plot(legend=True, ax=axis2, marker='o',rot=90, title="Percent Change")



# overlay customer data

median_cust.plot(legend=True, ax=axis1, marker='x', secondary_y=True)

pct_median_custchange.plot(legend=True, ax=axis2, marker='x', rot=90, secondary_y=True)
fig, (axis1, axis2, axis3, axis4, axis5) = plt.subplots(5, 1, sharex=True, figsize=(16,8))

# plot median sales

ax1 = median_sales.plot(legend=True, ax=axis1, marker='o',title="Median")

ax1.set_xticks(range(len(median_sales)))

ax1.set_xticklabels(median_sales.index.tolist(), rotation=90)

#pd.rolling_mean(median_sales, window=7).plot(ax=axis1)

# plot pct change

ax2 = pct_median_change.plot(legend=True, ax=axis2, marker='o',rot=90, title="Percent Change")



# overlay customer data

median_cust.plot(legend=True, ax=axis1, marker='x', secondary_y=True)

pct_median_custchange.plot(legend=True, ax=axis2, marker='x', rot=90, secondary_y=True)



# stateholiday overlay

# StateHoliday has a mixture of "O", 0 and "0", replace these with "O"

train.StateHoliday.replace(["O",0,"0"],['O','O','O'], inplace=True)

ax3 = sns.countplot(x='MonthYear', hue='StateHoliday', data=train[train['StateHoliday']!='O'], palette='husl', ax=axis3)

tmp = ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)

tmp = ax3.set_title('State Holidays')

#schoolholiday overlay

ax4 = sns.countplot(x='MonthYear', hue='SchoolHoliday', data=train[train['SchoolHoliday']!=0], palette='husl', ax=axis4)

subs = train[train['SchoolHoliday']!=0]

cntchange = subs.groupby('MonthYear')['SchoolHoliday'].count().pct_change()

cntchange.plot(ax=axis4, marker='x', secondary_y=True)

tmp = ax4.set_xticklabels(ax4.get_xticklabels(), rotation=90)

tmp = ax4.set_title('School Holidays and Percentage Change in School Holidays')

#promotions overlay

ax5 = sns.countplot(x='MonthYear', hue='Promo', data=train[train["Promo"]!=0], palette='husl', ax=axis5)

subs = train[train['Promo']!=0]

cntchange = subs.groupby('MonthYear')['Promo'].count().pct_change()

cntchange.plot(ax=axis5, marker='x', secondary_y=True)

tmp = ax5.set_xticklabels(ax5.get_xticklabels(), rotation=90)

tmp = ax5.set_title('Promotions and Percentage Change in Promotions')
# group sales/customer data by weekday

day = train[(train['Open']!=0)]

sales_day = day.groupby('DayOfWeek')['Sales'].median()

cust_day = day.groupby('DayOfWeek')['Customers'].median()

#

fig, (axis1) = plt.subplots(1,1, sharex=True, figsize=(10,5))

# plot median sales

ax1 = sales_day.plot(legend=True, ax=axis1, marker='o',title="Median")

ax1.set_xticks(sales_day.index)

tmp = ax1.set_xticklabels(sales_day.index.tolist(), rotation=90)

# overlay customer data

cust_day.plot(legend=True, ax=axis1, marker='x', secondary_y=True)
subs = train[train['Open']!=0] # select all data where the stores were open

# gather median and MAD for each store

sales_perstore = subs.groupby('Store')['Sales'].median()

mad_sales_perstore = subs.groupby('Store')['Sales'].mad()

cust_perstore = subs.groupby('Store')['Customers'].median()

mad_cust_perstore = subs.groupby('Store')['Customers'].mad()

# sales to customer ratio

sales_cust_ratio = sales_perstore/cust_perstore

mad_sales_cust_ratio = mad_sales_perstore/cust_perstore  # MAD for sales_cust_ratio is ratio between MAD sales and total customers

# plot scatter plot of sales vs customers

fig, (axis1) = plt.subplots(1,1, sharex=True,figsize=(15,7))

colors = np.random.rand(len(sales_perstore))

axis1.scatter(cust_perstore, sales_perstore, s=mad_sales_perstore, c=colors, cmap='jet', alpha=0.5)

axis1.set_xlabel('Customers')

axis1.set_ylabel('Sales')

axis1.set_title('Stores Sales with MAD auora')

for store in sales_perstore.index:

    #print('{}'.format(sales_perstore[i]))

    axis1.annotate(store, (cust_perstore[store], sales_perstore[store]))

# plot a fitted line

linefit = np.poly1d(np.polyfit(cust_perstore, sales_perstore, 1))(cust_perstore);

axis1.plot(cust_perstore, linefit)
subs = train[train['Open']!=0] # select all data where the stores were open

# sales to customer ratio

sales_cust_ratio = train['Sales']/train['Customers']  # will result in NaN's, look out

train['Sales-Customers'] = pd.Series(sales_cust_ratio, index=train.index)

subs_1 = train[train['Open']!=0] # select all data where the stores were open

subs = subs_1[subs_1['Sales-Customers'].notnull()] # remove NANs

# gather median and MAD for each store

SC_perstore = subs.groupby('Store')['Sales-Customers'].median()

mad_SC_perstore = subs.groupby('Store')['Sales-Customers'].mad()

cust_perstore = subs.groupby('Store')['Customers'].median()

mad_cust_perstore = subs.groupby('Store')['Customers'].mad()

# scatter plot of sales/customers vs customers

sns.set(font_scale=1)

fig, (axis1) = plt.subplots(1,1, sharex=True,figsize=(15,7))

colors = np.random.rand(len(SC_perstore))

axis1.scatter(cust_perstore, SC_perstore, s=mad_SC_perstore*1000, c=colors, cmap='jet', alpha=0.5) # multiplying "size" by 1000 to make the extent visible

axis1.set_xlabel('Customers')

axis1.set_ylabel('Sales/Customers')

axis1.set_title('Stores Sales/Customers ratio with MAD auora')

for store in SC_perstore.index:

    #print('{}'.format(sales_perstore[i]))

    axis1.annotate(store, (cust_perstore[store], SC_perstore[store]))
# create a temp dataframe

df = pd.DataFrame({'Sales/Customers ratio': SC_perstore, 'Customers':cust_perstore, 

                   'Sales/Customers-MAD': mad_SC_perstore, 'Store':SC_perstore.index})

# create a jointplot and add an joint density estimate

sns.set(font_scale=1)

ax1 = sns.jointplot(x='Customers', y='Sales/Customers ratio', data=df, 

                                   color='k', size=10)

ax1.plot_joint(sns.kdeplot, zorder=0, n_levels=30, cmap="Reds", shade=True, shade_lowest=False)



# annotate every datapoint with StoreID

for store in SC_perstore.index:

    #print('{}'.format(sales_perstore[i]))

    ax1.ax_joint.annotate(store, (df['Customers'][store], df['Sales/Customers ratio'][store]), fontsize=10)
# calculate kernel densities

from scipy import stats

values = np.vstack([cust_perstore.ravel(), SC_perstore.ravel()])

kde = stats.gaussian_kde(values)

density = kde(values)



# plot to confirm

sns.set(font_scale=1)

fig, (axis1,axis2, axis3) = plt.subplots(3,1, figsize=(10,10))

x,y = values

axis1.scatter(x,y, c=density)

axis1.set_title('kernel densities')

n, bins, patches = axis2.hist(density, 10)

axis2.set_title('histogram of densities')



#lowest threshold

thres = bins[1]-((bins[1] - bins[0])/2)



# plot the outlier points using the threshold

use_colors = {True:'red', False:'white'}

axis3.scatter(x,y, c = [use_colors[x <= thres] for x in density])

axis3.set_title('kernel densities with outlier annotated')

plt.show()



# outliers

outlier_stores = density < thres

df['Outlier'] = pd.Series(outlier_stores, index=df.index)



## Lets create the joint plot again and annotate only the outliers

## create a jointplot and add an joint density estimate

#sns.set(font_scale=3)

#ax1 = sns.jointplot(x='Customers', y='Sales/Customers ratio', data=df, 

#                                   color='k', size=30)

#ax1.plot_joint(sns.kdeplot, zorder=0, n_levels=30, cmap="Blues", shade=True, shade_lowest=False)

## annotate only outlier datapoint with StoreID

#for i, store in enumerate(SC_perstore.index):

#    #print('{}'.format(sales_perstore[i]))

#    if(outlier_stores[i]):

#        ax1.ax_joint.annotate(store, (df['Customers'][store], df['Sales/Customers ratio'][store]), fontsize=30, color='k')
# Lets create the joint plot again and annotate only the outliers

# create a jointplot and add an joint density estimate

sns.set(font_scale=1)

ax1 = sns.jointplot(x='Customers', y='Sales/Customers ratio', data=df, 

                                   color='k', size=10)

ax1.plot_joint(sns.kdeplot, zorder=0, n_levels=30, cmap="Blues", shade=True, shade_lowest=False)



#

sales_cust_ratio_threshold = [SC_perstore.mean() - 2*SC_perstore.std(), SC_perstore.mean() + 2*SC_perstore.std()] 

cust_threshold = [cust_perstore.mean() - 2*cust_perstore.std(), cust_perstore.mean() + 2*cust_perstore.std()]

# annotate only outlier datapoint with StoreID

outlier_annotate = ['n' for x in range(len(outlier_stores))] # creaet a array full of "normal(n)"

for i, store in enumerate(SC_perstore.index):

    #print('{}'.format(sales_perstore[i]))

    if(outlier_stores[i]):

        # color red if below threshold ELSE color green is above threshold

        if(df['Sales/Customers ratio'][store] <= sales_cust_ratio_threshold[0]

                or df['Customers'][store] >= cust_threshold[1]):

            outlier_annotate[i] = 'l' # low-performer

            ax1.ax_joint.annotate(store, (df['Customers'][store], df['Sales/Customers ratio'][store]), fontsize=10, color='r')

        

        elif (df['Sales/Customers ratio'][store] >= sales_cust_ratio_threshold[1]

                or df['Customers'][store] < cust_threshold[0]):

            outlier_annotate[i] = 'h' # high performer

            ax1.ax_joint.annotate(store, (df['Customers'][store], df['Sales/Customers ratio'][store]), fontsize=10, color='g')

        

        else:

            #outlier_annotate[i] = 'm' # medium performer

            ax1.ax_joint.annotate(store, (df['Customers'][store], df['Sales/Customers ratio'][store]), fontsize=10, color='k')

               

#

df['Annotation'] = pd.Series(outlier_annotate, index=df.index)

#



# performance measure

hp = df[df['Annotation'] == 'h']['Sales/Customers ratio'].median()

hc = df[df['Annotation'] == 'h']['Customers'].median()

lp = df[df['Annotation'] == 'l']['Sales/Customers ratio'].median()

lc = df[df['Annotation'] == 'l']['Customers'].median()

nop = df[df['Annotation'] == 'n']['Sales/Customers ratio'].median()

nc = df[df['Annotation'] == 'n']['Customers'].median()

print('High performers:  ${} - sales/customer,  {} - customers'.format(hp, hc))

print('Low performers:  ${} - sales/customer,  {} - customers'.format(lp, lc))

print('Normal performers:  ${} - sales/customer,  {} - customers'.format(nop, nc))
# load stores.csv

stores = pd.read_csv('../input/store.csv')

# merge previous stores data with new stores data

stores_update = stores.merge(df, how='outer', on='Store')



# extract stores-data for all three categories of performers

#high_performers = stores[stores['Store'].isin(high_stores)]

#low_stores = df[df['Annotation'] == 'l']['Store']

#low_performers = stores[stores['Store'].isin(low_stores)]

#normal_stores = df[df['Annotation'] == 'n']['Store']

#normal_performers = stores[stores['Store'].isin(normal_stores)]



# Storetype - distribution (normalised histograms) across High, Low and Normal performers 

#(creating histograms of category using another category as bins (not numeric) is a bit problematic so a long-winded approach)

# refer: http://themrmax.github.io/2015/11/13/grouped-histograms-for-categorical-data-in-pandas.html

storetype = stores_update.groupby('Annotation')['StoreType'].value_counts().sort_index()

s = storetype.unstack()

s[s.isnull()]=0

# for each annotation calculate normalised value_counts

for st in s.index:

    s.ix[st]  = s.ix[st]/s.ix[st].sum()

    

s[s.isnull()]=0  # convert all NAN to ZERO  

storetype = s





# Assortment - distribution (normalised histogram) across High, Low and Normal performers

assort = stores_update.groupby('Annotation')['Assortment'].value_counts().sort_index()

s = assort.unstack()

s[s.isnull()]=0

# for each annotation calculate normalised value_counts

for st in s.index:

    s.ix[st]  = s.ix[st]/s.ix[st].sum()

s[s.isnull()]=0  # convert all NAN to ZERO  

assort = s



# plot

sns.set(font_scale=1)

fig, (axis1, axis2) = plt.subplots(1,2, figsize=(14,5))

ax = storetype.plot(kind='bar', rot=45, ax=axis1)

tmp = ax.set_xticklabels(['High performers', 'Low performers', 'Normal performers'])

tmp = ax.set_title('Store types distribution for High, Low and Normal performing stores')

ax1 = assort.plot(kind='bar', rot=45, ax=axis2)

tmp = ax1.set_xticklabels(['High performers', 'Low performers', 'Normal performers'])

tmp = ax1.set_title('Assortment distribution for High, Low and Normal performing stores')
# Competition Distance - across High, Low and Normal performers

compdist = stores_update.groupby('Annotation')['CompetitionDistance'].median()

# plot

sns.set(font_scale=1.5)

fig, (axis1) = plt.subplots(1,1, figsize=(10,5))

ax = compdist.plot(kind='bar', rot=45, ax=axis1)

tmp = ax.set_xticklabels(['High performers', 'Low performers', 'Normal performers'])

tmp = ax.set_title('Competition Distance for High, Low and Normal performing stores')
# year and month when the competition opened

year = stores_update['CompetitionOpenSinceYear']

month = stores_update['CompetitionOpenSinceMonth']



# there are many NANs, remove them

stores_notnull = year.notnull() & month.notnull()

st = stores_update[stores_notnull]['Store']

stores_update['CompOpen'] = stores_update[stores_notnull]['CompetitionOpenSinceYear'].astype(int).astype(str).str.cat(stores_update[stores_notnull]['CompetitionOpenSinceMonth'].astype(int).astype(str).str.zfill(2), sep='-')



# extract data for stores with data for competition start dates

stores_withdata = train[train['Store'].isin(st)]

subs =[]

subs = stores_withdata[stores_withdata['Open'] != 0] # select all stores that were open





def get_periods(store):

    ''' For a given "store" extract competition open dates (try pd.Period as an error/null/nan-check)

    '''

    comp = stores_update[stores_update['Store']==store]['CompOpen'].tolist()[0]

    try:

        per = pd.Period(comp)

        return comp

    

    except:

        return None



def get_CompetitionOpenData_forStore(stores_data, store, buffer=1):

    ''' For a given "store", extract its relevant data based on competition dates

    '''

    # length of total period

    len_period = len(range(0,(2*buffer+1)))

    # get competition open date

    comp = get_periods(store)

    if comp is not None:

        # extract store data and return only the values (without the index)

        out = stores_data.loc[store, pd.Period(comp)-buffer:pd.Period(comp)+buffer].values

        # for some stores, data may not exist for the entire period, ignore those

        if out.size == len_period:

            #return out.ravel().tolist() # out is ndarray, need ravel to convert to 1-D array

            #print(store, len_period, len(out), out)

            return out.tolist()

        

        else:

            return None

        

    else:

        return None 

    

    

def get_data_forCompetitionOpen(selected_stores, attribute='Sales', buffer=1):

    ''' For a given attribute and buffer, extract data for all stores from selected_stores

    '''

    # get median sales for each month for each store

    s = subs.groupby(['Store','MonthYear'])[attribute].median()

    # create timeseries index

    per = pd.DatetimeIndex(s.index.get_level_values('MonthYear'), format='%Y-%m').to_period('m')

    # re-index the multiindex using Store and "per"

    #s.index.set_levels(per, level='MonthYear', inplace=True)

    new_index = list(zip(s.index.get_level_values('Store'), per))

    s.index = pd.MultiIndex.from_tuples(new_index, names=('Store', 'MonthYear'))

    

    # extract data for all stores in and around their respective competition dates

    #data = pd.concat([ get_CompetitionOpenData_forStore(s, store, 1) for store in stores_update['Store'][0:6]]) # to produce a indexed dataframe of extracted data

    #data = [get_CompetitionOpenData_forStore(s, store, buffer) for store in stores_update['Store'][0:6]]

    data = [get_CompetitionOpenData_forStore(s, store, buffer) for store in stores_update['Store']]

    # remove None from list

    data = [x for x in data if x is not None]

    return np.array(data)





buffer = 3

# Sales for competitionOpen

dataSales = get_data_forCompetitionOpen(selected_stores=subs, attribute='Sales', buffer=buffer)

# Customers for competitionOpen

dataCust = get_data_forCompetitionOpen(selected_stores=subs, attribute='Customers', buffer=buffer)
#dataCust = dataCust/dataCust.max(axis=1)

# plot

sns.set(font_scale=1)

fig, (axis1, axis2, axis3) = plt.subplots(3,1, sharex=True, figsize=(10,10))

ax1 = sns.tsplot(data=dataSales, ax=axis1, estimator=np.median, color='m')

ax1.set_xticks(range(0,2*buffer+1))

ax1.set_xticklabels(range(-buffer,buffer+1))

ax1.set_ylabel('Sales')

ax1.set_title('Sales drop in and around competition opening nearby')

ax2 = sns.tsplot(data=dataCust, ax=axis2, estimator=np.median,color='m')

ax2.set_title('Customers drop in and around competition opening nearby')

ax2.set_ylabel('Customers')

ax3 = sns.tsplot(data=dataSales/dataCust, ax=axis3, estimator=np.median, color='m')

ax3.set_xlabel('Competition Opened at 0')

ax3.set_ylabel('Sales/Customers')

ax3.set_title('Performance based on Sales/Customers in and around competition opening nearby')
import calendar

# select all stores that were open

subs = train[train['Open']!=0]

# groupby Year and Month

selected_sales = subs.groupby(['Year', 'Month'])['Sales'].median()

selected_cust = subs.groupby(['Year', 'Month'])['Customers'].median()



# plot

fig, (axis1) = plt.subplots(1,1, figsize=(10,7))

selected_sales.unstack().T.plot(ax=axis1)

tmp = axis1.set_title("Yearly Sales")

tmp = axis1.set_ylabel("Sales")

tmp = axis1.set_xticks(range(0,13))

tmp = axis1.set_xticklabels(calendar.month_abbr)