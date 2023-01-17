# learning from https://www.kaggle.com/robikscube/m5-forecasting-starter-data-exploration

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from itertools import cycle
pd.set_option('max_columns', 50)
plt.style.use('bmh')
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

cal = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')
train = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
submission = pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')
sellprice = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')

days = [d for d in train.columns if 'd_' in d]
# total sales over 1913 days of a product
train['total_sale'] = train.loc[:,days].sum(axis=1)

# sort items by sale
print(train['total_sale'].sort_values())
print('\n')


# index of an item
print(train.index[train.id == 'FOODS_3_090_CA_3_validation'][0])
print('\n')
# item most sold
print('Top 5:')
print(train.loc[8412,'id'].replace('_validation',''), ':', train.loc[8412,'total_sale'])
print(train.loc[18055,'id'].replace('_validation',''), ':', train.loc[18055,'total_sale'])
print(train.loc[21104,'id'].replace('_validation',''), ':', train.loc[21104,'total_sale'])
print(train.loc[8908,'id'].replace('_validation',''), ':', train.loc[8908,'total_sale'])
print(train.loc[2314,'id'].replace('_validation',''), ':', train.loc[2314,'total_sale'])

print('\n')

# item least sold
print('Bottom 5:')
print(train.loc[6682,'id'].replace('_validation',''), ':', train.loc[6682,'total_sale'])
print(train.loc[6048,'id'].replace('_validation',''), ':', train.loc[6048,'total_sale'])
print(train.loc[27606,'id'].replace('_validation',''), ':', train.loc[27606,'total_sale'])
print(train.loc[20192,'id'].replace('_validation',''), ':', train.loc[20192,'total_sale'])
print(train.loc[26276,'id'].replace('_validation',''), ':', train.loc[26276,'total_sale'])
# function to plot sale of any item, given 'id'
# matchs days in actual date
def plot_item(item_name):
        
    sale = train.loc[train.id == item_name, days].T # get daily sales, then transpose
    
    item_index = train.index[train.id == item_name][0] # index of this item
   
    item_name = item_name.replace('_validation','') # remove _validation
           
    sale = sale.rename(columns={item_index:item_name}) # rename column to item name
    sale = sale.reset_index().rename(columns={'index':'d'}) #re name column to d
    sale = sale.merge(cal, how='left', validate='1:1') # merge
    sale = sale.set_index('date')[item_name] # just keep the columns we need
    
    sale.plot(figsize=(22,9),lw=2)
    plt.title(f'{item_name} sale over time')
    plt.legend([item_name])
    plt.show()

# top 3
plot_item('FOODS_3_090_CA_3_validation')
# plot_item('FOODS_3_586_TX_2_validation')
# plot_item('FOODS_3_586_TX_3_validation')     

# bottom 3
#plot_item('HOUSEHOLD_1_020_CA_3_validation')
#plot_item('FOODS_3_778_CA_2_validation')
#plot_item('HOBBIES_1_170_WI_3_validation')
# function to plot sale of any item, given 'id'
# by week/month/year

def plot_by_wmy(item_name):

    # one row, 3 column
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(16,3))
    
    sale = train.loc[train.id == item_name, days].T # get daily sales, then transpose

    item_index = train.index[train.id == item_name][0] # index of this item

    item_name = item_name.replace('_validation','') # remove _validation

    sale = sale.rename(columns={item_index:item_name}) # rename column to item name
    sale = sale.reset_index().rename(columns={'index':'d'}) #re name column to d
    sale = sale.merge(cal, how='left', validate='1:1') # merge

    sale_day = sale.groupby('wday')[item_name].mean() # mean sale by day of week
    sale_month = sale.groupby('month')[item_name].mean() # mean sale by day of week
    sale_year = sale.groupby('year')[item_name].mean() # mean sale by day of week

    sale_day.plot(kind='line',lw=8,title='average sale, day of week',color='blue',ax=ax1) # plot by day
    sale_month.plot(kind='line',lw=8,title='average sale, by month', color='red',ax=ax2) # plot by month
    sale_year.plot(kind='line',lw=8,title='average sale, by year',color='green',ax=ax3) # plot by year
    
    ax1.set_xlabel('day')

    fig.suptitle(f'Trend for: {item_name}' ,size=20,y=1.2) # title

    plt.show()

plot_by_wmy('FOODS_3_090_CA_3_validation')
plot_by_wmy('FOODS_3_586_TX_2_validation')
plot_by_wmy('FOODS_3_586_TX_3_validation')
# total number of goods in each category

a = pd.DataFrame(train.groupby('cat_id').count()['id'])
a = a.rename(columns={'id':'Count'})

a.plot.pie(y='Count', figsize=(10,10))
plt.legend(loc='upper left')
plt.show()
# Sale of each Category each day

a =train.set_index('id')[days].T  # d, item1, item2 ...
b =cal.set_index('d')['date']     # d, date

past_sales = a.merge(b, left_index=True, right_index=True, validate='1:1').set_index('date') # date, item1, item2 ...


items_col = [c for c in past_sales.columns if 'HOBBIES' in c]
past_sales[items_col].sum(axis=1).plot(kind='line',figsize=(20,5))

items_col = [c for c in past_sales.columns if 'FOOD' in c]
past_sales[items_col].sum(axis=1).plot(kind='line',figsize=(20,5))

items_col = [c for c in past_sales.columns if 'HOUSEHOLD' in c]
past_sales[items_col].sum(axis=1).plot(kind='line',figsize=(20,5))

plt.title('Sale by category',y=1.1, size=25)
plt.legend(['HOBBIES','FOOD','HOUSEHOLD'])

plt.show()
# rolling mean sales by store
store_list = train.store_id.unique()
store_list

a = train.groupby('store_id')[days].sum().T # d, store
b =cal.set_index('d')['date']     # d, date

store_sales = a.merge(b, left_index=True, right_index=True, validate='1:1').set_index('date') # date, store


# rolling 30 days
plt.figure(figsize=(24,5))

for store in store_list:
    
    store_sales[store].rolling(30).mean().plot(kind='line')

plt.title('Rolling 30 days sale for all stores')
plt.legend(store_list)
plt.show()


# rolling 90 days
plt.figure(figsize=(24,5))

for store in store_list:
    
    store_sales[store].rolling(90).mean().plot(kind='line')

plt.title('Rolling 90 days sale for all stores')
plt.legend(store_list)
plt.show()
# Rolling 7 day sales by store

fig, axes = plt.subplots(5,2,figsize=(25,10), sharex=True)

axes = axes.flatten()

ax_idx = 0

for store in store_list:
    
    store_sales[store].rolling(7).mean().plot(kind='line',title=store, lw = 2, color=next(color_cycle), ax=axes[ax_idx])
    
    ax_idx += 1

plt.suptitle('Rolling 7 days sale by store', size=30, y=1.2)
plt.tight_layout()
plt.show()
# sort days by sales, min to max
a = pd.DataFrame(store_sales.sum(axis=1).sort_values(),columns=['Sale'])
a
# preping for heatmap 52weeks

a = train.groupby('cat_id')[days].sum().T # d, store
b = cal.set_index('d').loc[:,['date','wday','month','weekday']]     # d, date

store_sales = a.merge(b, left_index=True, right_index=True, validate='1:1').set_index('date') # date, store

store_sales.index = pd.to_datetime(store_sales.index) # make index date format

# everyday sale by weekdays, (FOODS HOBBIES HOUSEHOLD), (start date), (year), rolling 364 day

def heatmap_52weeks(cat_id,start_date,year):
    
    # pick the dates, then reset index so can add columns in new DF
    sale = store_sales.loc[store_sales.index.isin(pd.date_range(start=start_date, periods=364))]
    sale = sale.reset_index()
    
    # new df used for plot
    # intialize a DF
    sale_week = pd.DataFrame({'day':['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],'week 1':sale[cat_id][0:7]})

    # add columns week 2:52
    for i in range(1,52):
         sale_week[f'week {i+1}'] = sale.loc[i*7 : i*7+7,cat_id].reset_index()[cat_id]

    
    sale_week = sale_week.set_index('day')
 
    plt.figure(figsize=(25,5))
    sns.heatmap(sale_week, square=True, cmap='seismic')
    plt.title(f'{cat_id}: Year {year}')
    plt.show()

# food 
heatmap_52weeks('FOODS','2012-01-02', '2012')
heatmap_52weeks('FOODS','2012-12-31', '2013')
heatmap_52weeks('FOODS','2013-12-30', '2014')
heatmap_52weeks('FOODS','2014-12-29', '2015')
# hobby
heatmap_52weeks('HOBBIES','2012-01-02', '2012')
heatmap_52weeks('HOBBIES','2012-12-31', '2013')
heatmap_52weeks('HOBBIES','2013-12-30', '2014')
heatmap_52weeks('HOBBIES','2014-12-29', '2015')
# household
heatmap_52weeks('HOUSEHOLD','2012-01-02', '2012')
heatmap_52weeks('HOUSEHOLD','2012-12-31', '2013')
heatmap_52weeks('HOUSEHOLD','2013-12-30', '2014')
heatmap_52weeks('HOUSEHOLD','2014-12-29', '2015')
store_list = train.store_id.unique()
store_list

# function that produce trend of price over time, given 'item name', list of stores

def price_over_time(item_name,store_names):
    fig, ax = plt.subplots(figsize=(22, 5))
    
    for SS in store_names:

        prices = sellprice.loc[sellprice.item_id == item_name].loc[sellprice.store_id == SS]
        prices = prices.loc[:,['wm_yr_wk','sell_price']]

        prices.plot(x='wm_yr_wk',
                  y='sell_price',
                  style='.',
                  color=next(color_cycle),
                  figsize=(15, 5),
                  title=f'{item_name} sale price over time',
                  ax=ax
                )
    
    plt.legend(store_names)
    plt.show()

price_over_time('FOODS_3_586',['TX_1','TX_2','TX_3'])
# preping for holiday effect

a = train.groupby('cat_id')[days].sum().T # d, store
b = cal.set_index('d').loc[:,['date','wday','month','weekday','event_name_1','event_type_1']] # d, date

store_sales = a.merge(b, left_index=True, right_index=True, validate='1:1').set_index('date') # date, store

store_sales.index = pd.to_datetime(store_sales.index) # make index date format

ss = store_sales

# list all of holiday, remove NaN
events = ss.event_name_1.unique()
events = events[1:len(events)]
events
# FOOD sale on holiday
fig, axes = plt.subplots(10,3,figsize=(25,60))

axes = axes.flatten()
ax_idx = 0


for ee in events:
    
    normal_day_sales = ss.loc[(ss.event_name_1.isnull()) & (ss.weekday != 'Saturday') & (ss.weekday != 'Sunday')] # sales on normal day
    nm = normal_day_sales['FOODS'].mean()  

    event_day = ss.loc[(ss.event_name_1 == ee)] # sale on the holiday
    em = event_day['FOODS'].mean()

    df = pd.DataFrame({'Type':[ee,'Normal day'],'Sale':[em,nm]}) # make new df
    
    df.plot(kind='bar',x='Type',y='Sale', title= ee, legend='', rot=0, color=['green','black'], ax=axes[ax_idx])
    
    axes[ax_idx].set_title(ee, fontsize=20)
    axes[ax_idx].tick_params(axis='both', which='major', labelsize=20) # x y tick font size
    axes[ax_idx].set_xlabel('') # remove x label
    axes[ax_idx].grid(False) # remove grid
    
    ax_idx +=1
    
plt.tight_layout()
plt.suptitle('Food Sales on Holidays', size=40, y=1.02)
plt.subplots_adjust(hspace=0.25, wspace=0.25)
plt.show()
    
    
# Hobbies sale on holiday
fig, axes = plt.subplots(10,3,figsize=(25,60))

axes = axes.flatten()
ax_idx = 0

for ee in events:
    
    normal_day_sales = ss.loc[(ss.event_name_1.isnull()) & (ss.weekday != 'Saturday') & (ss.weekday != 'Sunday')] # sales on normal day
    nm = normal_day_sales['HOBBIES'].mean()  

    event_day = ss.loc[(ss.event_name_1 == ee)] # sale on the holiday
    em = event_day['HOBBIES'].mean()

    df = pd.DataFrame({'Type':[ee,'Normal day'],'Sale':[em,nm]}) # make new df
    
    df.plot(kind='bar',x='Type',y='Sale', title= ee, legend='', rot=0, color=['dodgerblue','black'], ax=axes[ax_idx])
    
    axes[ax_idx].set_title(ee, fontsize=20)
    axes[ax_idx].tick_params(axis='both', which='major', labelsize=20) # x y tick font size
    axes[ax_idx].set_xlabel('') # remove x label
    axes[ax_idx].grid(False) # remove grid
    
    ax_idx +=1
    
plt.tight_layout()
plt.suptitle('Hobbies Sales on Holidays', size=40, y=1.02)
plt.subplots_adjust(hspace=0.25, wspace=0.25)
plt.show()
    
    
# Household sale on holiday
fig, axes = plt.subplots(10,3,figsize=(25,60))

axes = axes.flatten()
ax_idx = 0

for ee in events:
    
    normal_day_sales = ss.loc[(ss.event_name_1.isnull()) & (ss.weekday != 'Saturday') & (ss.weekday != 'Sunday')] # sales on normal day
    nm = normal_day_sales['HOUSEHOLD'].mean()  

    event_day = ss.loc[(ss.event_name_1 == ee)] # sale on the holiday
    em = event_day['HOUSEHOLD'].mean()

    df = pd.DataFrame({'Type':[ee,'Normal day'],'Sale':[em,nm]}) # make new df
    
    df.plot(kind='bar',x='Type',y='Sale', title= ee, legend='', rot=0, color=['peru','black'], ax=axes[ax_idx])

    axes[ax_idx].set_title(ee, fontsize=20)
    axes[ax_idx].tick_params(axis='both', which='major', labelsize=20) # x y tick font size
    axes[ax_idx].set_xlabel('') # remove x label
    axes[ax_idx].grid(False) # remove grid
    
    ax_idx +=1
    
plt.tight_layout()
plt.suptitle('Household Sales on Holidays', size=40, y=1.02)
plt.subplots_adjust(hspace=0.25, wspace=0.25)
plt.show()
    

# event types, 'Sporting', 'Cultural', 'National', 'Religious'
event_types = ss.event_type_1.unique()
event_types = event_types[1:len(event_types)]
# Food sale on type holiday
fig, axes = plt.subplots(2,2,figsize=(10,10))

axes = axes.flatten()
ax_idx = 0

for ep in event_types:
    
    normal_day_sales = ss.loc[(ss.event_type_1.isnull()) & (ss.weekday != 'Saturday') & (ss.weekday != 'Sunday')] # sales on normal day
    nm = normal_day_sales['FOODS'].mean()  

    event_day = ss.loc[(ss.event_type_1 == ep)] # sale on the holiday
    em = event_day['FOODS'].mean()

    df = pd.DataFrame({'Type':[ee,'Normal day'],'Sale':[em,nm]}) # make new df
    
    df.plot(kind='bar',x='Type',y='Sale', title= ep, legend='', rot=0, color=['green','black'], ax=axes[ax_idx])
    
    axes[ax_idx].set_xlabel('') # remove x label
    axes[ax_idx].grid(False) # remove grid
    
    ax_idx +=1
    
plt.tight_layout()
plt.suptitle('Food Sales on Holidays', size=30, y=1.05)
plt.subplots_adjust(hspace=0.25, wspace=0.25)
plt.show()
    

# Hobby sale on type holiday
fig, axes = plt.subplots(2,2,figsize=(10,10))

axes = axes.flatten()
ax_idx = 0

for ep in event_types:
    
    normal_day_sales = ss.loc[(ss.event_type_1.isnull()) & (ss.weekday != 'Saturday') & (ss.weekday != 'Sunday')] # sales on normal day
    nm = normal_day_sales['HOBBIES'].mean()  

    event_day = ss.loc[(ss.event_type_1 == ep)] # sale on the holiday
    em = event_day['HOBBIES'].mean()

    df = pd.DataFrame({'Type':[ee,'Normal day'],'Sale':[em,nm]}) # make new df
    
    df.plot(kind='bar',x='Type',y='Sale', title= ep, legend='', rot=0, color=['dodgerblue','black'], ax=axes[ax_idx])
    
    axes[ax_idx].set_xlabel('') # remove x label
    axes[ax_idx].grid(False) # remove grid
    
    ax_idx +=1
    
plt.tight_layout()
plt.suptitle('Hobbies Sales on Holidays', size=30, y=1.05)
plt.subplots_adjust(hspace=0.25, wspace=0.25)
plt.show()
    

# Household sale on type holiday
fig, axes = plt.subplots(2,2,figsize=(10,10))

axes = axes.flatten()
ax_idx = 0

for ep in event_types:
    
    normal_day_sales = ss.loc[(ss.event_type_1.isnull()) & (ss.weekday != 'Saturday') & (ss.weekday != 'Sunday')] # sales on normal day
    nm = normal_day_sales['HOUSEHOLD'].mean()  

    event_day = ss.loc[(ss.event_type_1 == ep)] # sale on the holiday
    em = event_day['HOUSEHOLD'].mean()

    df = pd.DataFrame({'Type':[ee,'Normal day'],'Sale':[em,nm]}) # make new df
    
    df.plot(kind='bar',x='Type',y='Sale', title= ep, legend='', rot=0, color=['peru','black'], ax=axes[ax_idx])
    
    axes[ax_idx].set_xlabel('') # remove x label
    axes[ax_idx].grid(False) # remove grid
    
    ax_idx +=1
    
plt.tight_layout()
plt.suptitle('Household Sales on Holidays', size=30, y=1.05)
plt.subplots_adjust(hspace=0.25, wspace=0.25)
plt.show()
    
