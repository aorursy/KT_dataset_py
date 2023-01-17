import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime

import copy

import matplotlib as mpl

from statsmodels.tsa.seasonal import seasonal_decompose

from dateutil.parser import parse

import statsmodels.api as sm

from sklearn.metrics import mean_squared_error

from math import sqrt
train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")

sample_submission = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")
train.head(3)
train.describe()
test.head()
items.head()
item_categories.head()
shops.head()
test_shops_uniq = test['shop_id'].unique()

train_shops_uniq = train['shop_id'].unique()

# shops in test which is not in train

test_shops_uniq[np.logical_not(pd.Series(test_shops_uniq).isin(train_shops_uniq))]
test_item_uniq = test['item_id'].unique()

train_item_uniq = train['item_id'].unique()

# items in test which is not in train

test_items_is_not_in_train = test_item_uniq[np.logical_not(pd.Series(test_item_uniq).isin(train_item_uniq))]

test_items_is_not_in_train
train.info()
# let's do the date column in correct format

train['date']=train['date'].apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))
train.info()
monthly_sales = train.groupby(["shop_id","item_id","date_block_num"])[

    "date","item_price","item_cnt_day"].agg({"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})
monthly_sales.head(20)
train['item_sales_day'] = train.item_cnt_day*train.item_price
# Function for creating DataFrame with days or months sales

def sales_df (period):

    time = train.copy()

    time.set_index(['date'],inplace=True)

    time_1 = time['item_cnt_day'].resample(period).sum()

    df = pd.DataFrame(time_1)

    time_2 = time['item_sales_day'].resample(period).sum()

    df['item_sales_day'] = time_2

    time_3 = time['item_price'].resample(period).mean()

    df['item_price'] = time_3

    return df
df_months = sales_df (period='M')
df_days = sales_df (period='D')
df_months.to_excel('df_months.xlsx')
df_days.to_excel('df_days.xlsx')
# Draw Plot

def plot_df(df, x, y, title="", xlabel='Time', ylabel='Value', dpi=100):

    plt.figure(figsize=(20,5), dpi=dpi)

    plt.plot(x, y, color='tab:red')

    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)

    plt.show()
plot_df(df_months, x=df_months.index, y=df_months.item_sales_day.values, 

        title='Total Sales (money) of the company by months') 
plot_df(df_months, x=df_days.index, y=df_days.item_sales_day.values, 

        title='Total Sales (money) of the company by days') 
plot_df(df_months, x=df_months.index, y=df_months.item_cnt_day.values, 

        title='Total Sales (amount of products) of the company by months') 
plot_df(df_months, x=df_days.index, y=df_days.item_cnt_day.values, 

        title='Total Sales (amount of products) of the company by days') 
plot_df(df_months, x=df_months.index, y=df_months.item_price.values, 

        title='Mean prices of the company by months') 
plot_df(df_months, x=df_days.index, y=df_days.item_price.values, 

        title='Mean prices of the company by days') 
df_months_season = df_months.copy()
df_months_season['year'] = [d.year for d in df_months_season.index]

df_months_season['month'] = [d.month for d in df_months_season.index]
years = df_months_season['year'].unique()

years
df_months_season.loc[df_months_season.year==2013, :]
plt.plot('month', 'item_sales_day', data=df_months_season.loc[df_months_season.year==2013, :], label='2013')
def seasonal_plot(df, title, xlabel, ylabel, feature):

    # Prep Colors

    np.random.seed(100)

    mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)



    # Draw Plot

    plt.figure(figsize=(16,12), dpi= 80)

    for i, y in enumerate(years):

            plt.plot('month', feature, data=df.loc[df.year==y, :], color=mycolors[i], label=y)

            plt.text(df.loc[df.year==y, :].shape[0]-.9, df.loc[df.year==y, feature][-1:].values[0], y, 

                     fontsize=12, color=mycolors[i])



    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)

    plt.show()
seasonal_plot(df_months_season, title = 'Seasonal Plot of Total Sales (money)', xlabel='Month', ylabel='Sales (money)', 

              feature='item_sales_day')
seasonal_plot(df_months_season, title = 'Seasonal Plot of Mean Price', xlabel='Month', ylabel='Price', 

              feature='item_price')
seasonal_plot(df_months_season, title = 'Seasonal Plot of Total Sales (amount of products)', xlabel='Month', 

              ylabel='Sales (amount of product)', 

              feature='item_cnt_day')
df_days_season = df_days.copy()
df_days_season['year'] = [d.year for d in df_days_season.index]

df_days_season['month'] = [d.month for d in df_days_season.index]
years = df_days_season['year'].unique()

years
# Multiplicative Decomposition 

def mult_decomposition (df, v):

    result_mul = sm.tsa.seasonal_decompose(df[v], extrapolate_trend='freq', model="multiplicative")

    return result_mul
# Additive Decomposition

def add_decomposition (df, v):

    result_add = sm.tsa.seasonal_decompose(df[v], model='additive', extrapolate_trend='freq')

    return result_add
# Plot decomposition

def plot_decomposition (result_mul, result_add):

    plt.rcParams.update({'figure.figsize': (10,10)})

    result_mul.plot().suptitle('Multiplicative Decompose', fontsize=20)

    result_add.plot().suptitle('Additive Decompose', fontsize=20)

    plt.show()
result_mul_days = mult_decomposition (df=df_days, v='item_sales_day')

result_add_days = add_decomposition (df=df_days, v='item_sales_day')

plot_decomposition (result_mul_days, result_add_days)
# Extract the Components ----

# Actual Values = Product of (Seasonal * Trend * Resid)

def extract_compon(result):

    df_reconstructed = pd.concat([pd.Series(result.seasonal), pd.Series(result.trend),

                              pd.Series(result.resid), pd.Series(result.observed)], axis=1)

    df_reconstructed.columns = ['seas', 'trend', 'resid', 'actual_values']

    return df_reconstructed
extract_compon(result_add_days)
result_mul_months = mult_decomposition (df=df_months, v='item_sales_day')

result_add_months = add_decomposition (df=df_months, v='item_sales_day')

plot_decomposition (result_mul_months, result_add_months)
extract_compon(result_add_months)
# creating train table with sales by months



sales_by_months = pd.pivot_table(train,

                                 values=['item_cnt_day', 'item_price', 'item_sales_day'],

                                 index=['shop_id', 'item_id'],

                                 columns = ['date_block_num'],

                                 aggfunc={'item_cnt_day':np.sum, 'item_price':np.mean, 'item_sales_day':np.sum})

sales_by_months = sales_by_months.fillna(0)

sales_by_months = sales_by_months.reset_index()



# adding item_category_id column



m_sales_by_months = pd.merge(sales_by_months, items[['item_id', 'item_category_id']], how = 'left', on=['item_id'])
sales_by_months.head()
m_sales_by_months.head()
test.head()
# adding information from train table (sales in amount, sales in money, price of item)



t = test.copy()

m_test = pd.merge(t, sales_by_months, how = 'left', on=['shop_id', 'item_id'])



# adding item_category_id column



m_test = pd.merge(m_test, items[['item_id', 'item_category_id']], how = 'left', on=['item_id'])

m_test.head()
a = []

for i in range(0,34):

    a.append(('item_cnt_day', i))

for i in range(0,34):

    a.append(('item_sales_day', i))

for i in range(0,34):

    a.append(('item_price', i))

    

category_sales = pd.DataFrame(columns = a)

for j in m_sales_by_months['item_category_id'].value_counts().index:

    category_sales = category_sales.append(m_sales_by_months[a][m_sales_by_months['item_category_id']==j].mean(),

                                           ignore_index=True)

category_sales.index = m_sales_by_months['item_category_id'].value_counts().index

category_sales = category_sales.reset_index()

category_sales = category_sales.rename(columns = {'index':"item_category_id"})

category_sales.head()
notnull_test = m_test[m_test[('item_cnt_day', 0)].notnull()]

null_test_categories_without_shops = pd.merge(m_test[m_test[('item_cnt_day', 

                                                             0)].isnull()][['ID',

                                                                                'shop_id', 

                                                                                'item_id',

                                                                                'item_category_id']],

                     category_sales, how = 'left', on=['item_category_id'],

                     copy = False).dropna(axis=1)

null_test_categories_without_shops.index = null_test_categories_without_shops['ID']
null_test_categories_without_shops.head()
null_test_categories_without_shops.info()
test_categories_without_shops = m_test.copy()

test_categories_by_shops = m_test.copy()
for i in test_categories_without_shops.columns:

    test_categories_without_shops.loc[

        test_categories_without_shops.ID.isin(null_test_categories_without_shops.ID),

                                      i] = null_test_categories_without_shops[i]
test_categories_without_shops.head()
test_categories_without_shops.info()
category_sales_by_shops = pd.DataFrame(columns = a)

shop = pd.DataFrame(columns = ['shop_id'])

category = pd.DataFrame(columns = ['item_category_id'])

for i in m_sales_by_months[('shop_id','')].value_counts().index:

    for j in m_sales_by_months['item_category_id'].value_counts().index:

        category_sales_by_shops = category_sales_by_shops.append(

            m_sales_by_months[a][(m_sales_by_months['item_category_id'

                                                   ]==j)&(m_sales_by_months[('shop_id', '')]==i)].mean(),

                                           ignore_index=True)

        shop = shop.append(pd.DataFrame([[i]], columns = ['shop_id']))

        category = category.append(pd.DataFrame([[j]], columns = ['item_category_id']))

shop = shop.reset_index()
shop = shop.drop(['index'], axis = 1)
category = category.reset_index()

category = category.drop(['index'], axis = 1)
category_sales_by_shops['shop_id'] = shop

category_sales_by_shops['item_category_id'] = category
category_sales_by_shops.head()
null_test_categories_by_shops = pd.merge(m_test[m_test[('item_cnt_day', 0)].isnull()][['ID', 'shop_id', 'item_id',

                                                                                            'item_category_id']],

                     category_sales_by_shops, 

                                         how = 'left',

                                         on = ['shop_id', 'item_category_id']

                     )

null_test_categories_by_shops.index = null_test_categories_by_shops['ID']
null_test_categories_by_shops.head()
for i in test_categories_by_shops.columns:

    test_categories_by_shops.loc[test_categories_by_shops.ID.isin(null_test_categories_by_shops.ID),

                                      i] = null_test_categories_by_shops[i]
test_categories_by_shops.head()
def predicted_amount(df, month):

    infl = ((df['item_price'][month-1]-df['item_price'][month-13])/df['item_price'][month-13])/12

    infl = infl.fillna(0)

    df['item_sales_day', '%s %d' % ('predicted', month)] = ((df['item_sales_day'][month-13]+

                                          df['item_sales_day'][month-1])/2)*(infl+1)

    df['item_price', '%s %d' % ('predicted', month)] = (infl+1)*df['item_price'][month-1]

    df['item_cnt_days', '%s %d' % ('predicted', month)] = df[

        'item_sales_day', '%s %d' % ('predicted', month)]/df['item_sales_day', '%s %d' % ('predicted', month)]

    df = df.fillna(0)

    return df
# def predicted_amount1(df, month):

#     df['item_cnt_days', '%s %d' % ('predicted', month)] = df['item_cnt_day'][month-1]

#     return df
sales_by_months.head()
# pr_sales_by_months1 = predicted_amount1(sales_by_months, 33)
pr_sales_by_months = predicted_amount(sales_by_months, 33)
pr_sales_by_months1.head()
rmse1 = sqrt(mean_squared_error(pr_sales_by_months['item_cnt_day'][33], 

                                pr_sales_by_months['item_cnt_days']['predicted 33']))
rmse1
def predicted_amount_test(df, month):

    infl = ((df['item_price', month-1]-df['item_price', month-13])/df['item_price', month-13])/12

    infl = infl.fillna(0)

    

    df['infl', '%s %d' % ('predicted', month)] = infl

    

    df['item_sales_day', '%s %d' % ('predicted', month)] = np.nan

    df['item_sales_day', '%s %d' % ('predicted', month)][infl!=np.inf] = ((df['item_sales_day', month-13][infl!=np.inf]+

                                          df['item_sales_day', month-1][infl!=np.inf])/2)*(infl[infl!=np.inf]+1)

    df['item_sales_day', '%s %d' % ('predicted', month)][infl==np.inf] = df['item_sales_day', month-1][infl==np.inf]

    

    

    df['item_price', '%s %d' % ('predicted', month)] = np.nan

    df['item_price', '%s %d' % ('predicted', month)][infl!=np.inf] = (infl[infl!=np.inf]+

                                                                      1)*df['item_price', month-1][infl!=np.inf]

    df['item_price', '%s %d' % ('predicted', month)][infl==np.inf] = df['item_price', month-1][infl==np.inf]

    

    

    df['item_cnt_days', '%s %d' % ('predicted', month)] = 0

    df['item_cnt_days', '%s %d' % ('predicted', month)][df['item_price', '%s %d' % ('predicted', month)]!=0] = df[

        'item_sales_day', '%s %d' % ('predicted', month)][df[

        'item_price', '%s %d' % ('predicted', month)]!=0]/df['item_price', '%s %d' % ('predicted', month)][df[

        'item_price', '%s %d' % ('predicted', month)]!=0]

    df = df.fillna(0)

    return df
pr_test_categories_without_shops = test_categories_without_shops.copy()
pr_test_categories_without_shops = predicted_amount_test(pr_test_categories_without_shops, 33)
pr_test_categories_without_shops.head()
rmse_test_1 = sqrt(mean_squared_error(pr_test_categories_without_shops[('item_cnt_day', 33)],

                                      pr_test_categories_without_shops[('item_cnt_days','predicted 33')]))

rmse_test_1
sample_submission.head()
pr_test_categories_without_shops_34 = predicted_amount_test(test_categories_without_shops, 34)
pr_test_categories_without_shops_34.head()
submission_1 = pr_test_categories_without_shops_34[['ID', ('item_cnt_days', 'predicted 34')]]

submission_1 = submission_1.rename(columns = {('item_cnt_days', 'predicted 34'):'item_cnt_month'})

# submission_1
submission_1.to_csv('submission_1.csv', index=False)
submission = pd.read_csv('submission_1.csv')
pr_test_categories_by_shops = test_categories_by_shops.copy()
pr_test_categories_by_shops = predicted_amount_test(pr_test_categories_by_shops, 33)
rmse_test_2 = sqrt(mean_squared_error(pr_test_categories_by_shops[('item_cnt_day', 33)],

                                      pr_test_categories_by_shops[('item_cnt_days','predicted 33')]))

rmse_test_2
pr_test_categories_by_shops_34 = predicted_amount_test(test_categories_by_shops, 34)
submission_2 = pr_test_categories_by_shops_34[['ID',('item_cnt_days', 'predicted 34')]]

submission_2 = submission_2.rename(columns = {('item_cnt_days', 'predicted 34'):'item_cnt_month'})
submission_2['item_cnt_month'] = submission_2['item_cnt_month'].clip(0,20)
submission_2.to_csv('submission_2.csv', index=False)