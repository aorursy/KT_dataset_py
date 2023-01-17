#Importing all the necessary libraries



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import datetime as dt



%matplotlib inline
#Plotly Libraries



import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.colors import n_colors

from plotly.subplots import make_subplots
olist_orders_dataset = pd.read_csv('../input/brazilian-ecommerce/olist_orders_dataset.csv')

olist_orders_dataset.head(10)
olist_sellers_dataset = pd.read_csv('../input/brazilian-ecommerce/olist_sellers_dataset.csv')

olist_sellers_dataset.head(10)
olist_customers_dataset = pd.read_csv('../input/brazilian-ecommerce/olist_customers_dataset.csv')

olist_customers_dataset.head(10)
olist_products_dataset = pd.read_csv('../input/brazilian-ecommerce/olist_products_dataset.csv')

olist_products_dataset.head(10)
olist_order_payments_dataset = pd.read_csv('../input/brazilian-ecommerce/olist_order_payments_dataset.csv')

olist_order_payments_dataset.head(10)
olist_geolocation_dataset = pd.read_csv('../input/brazilian-ecommerce/olist_geolocation_dataset.csv')

olist_geolocation_dataset.head(10)
olist_order_reviews_dataset = pd.read_csv('../input/brazilian-ecommerce/olist_order_reviews_dataset.csv')

olist_order_reviews_dataset.head(10)
olist_order_items_dataset = pd.read_csv('../input/brazilian-ecommerce/olist_order_items_dataset.csv')

olist_order_items_dataset.head(10)
product_category_name_translation = pd.read_csv('../input/brazilian-ecommerce/product_category_name_translation.csv')

product_category_name_translation.head(10)
df = olist_orders_dataset.merge(olist_order_reviews_dataset, on='order_id', how='left')

df = df.merge(olist_order_payments_dataset, on='order_id', how='left')

df_1 = olist_order_items_dataset.merge(olist_sellers_dataset, on='seller_id', how='left')

df_1 = df_1.merge(olist_products_dataset, on='product_id', how='left')

df = df.merge(df_1, on='order_id', how='left')

df = df.merge(olist_customers_dataset, on='customer_id', how='left')
df.isna().sum()
df.drop(['review_comment_title','review_comment_message', 'review_creation_date','review_answer_timestamp', 'shipping_limit_date', 'product_weight_g',

       'product_length_cm', 'product_height_cm', 'product_width_cm', 'order_delivered_carrier_date',

       'order_delivered_customer_date', 'order_estimated_delivery_date',

       'order_approved_at'], axis=1, inplace = True)
plt.figure(figsize=(16,9))

sns.heatmap(df.isna(), cmap="viridis")
df.drop_duplicates(inplace=True)
values = {'product_category_name': 0, 'product_name_lenght': 0, 'product_description_lenght': 0, 'product_photos_qty': 0}

df.fillna(value=values, inplace=True)
df_canc = df[df['order_status'] == 'canceled']
df_unavailable = df[df['order_status'] == 'unavailable']
pd.DataFrame(df[df['order_item_id'].isna()]['order_status'].value_counts())
df.dropna(subset=['order_item_id'], inplace=True)
df.dropna(subset=['payment_sequential'], inplace=True)
df.info()
#df.to_csv('final_olist.csv', index = False, sep=',', encoding='utf-8')
#1. Frequency of purchases per customer.

pd.DataFrame(df.groupby(['customer_id'])['order_id'].count().sort_values(ascending = False))
#2. Time series analysis of orders. Use of DateTime library in python to separate date and time from the necessary columns.

df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['year_of_purchase'] = pd.DatetimeIndex(df['order_purchase_timestamp']).year

df['month_of_purchase'] = pd.DatetimeIndex(df['order_purchase_timestamp']).month

df['date_of_purchase'] = pd.DatetimeIndex(df['order_purchase_timestamp']).day

df['time_of_purchase'] = pd.DatetimeIndex(df['order_purchase_timestamp']).hour
monthly_orders = pd.DataFrame(df.groupby(['month_of_purchase'])['month_of_purchase'].count())
monthly_orders = monthly_orders.rename(index={1: 'January', 2: 'February', 3: 'March', 4: 'April', 5:'May', 6:'June', 7:'July',8:'August', 9:'September', 10:'October', 11:'November', 12:'December'})
monthly_orders
fig = go.Figure(data=go.Scatter(x=monthly_orders.index,

                                y=monthly_orders['month_of_purchase'],

                                mode='lines+markers')) # hover text goes here

fig.update_layout(title='Months With Most Orders',xaxis_title="Month",yaxis_title="Number of Orders")

fig.update_yaxes(type="log")

fig.show()
monthly_analysis_jan = pd.DataFrame(df[df['month_of_purchase'] == 1].groupby(['date_of_purchase'])['date_of_purchase'].count())

monthly_analysis_feb = pd.DataFrame(df[df['month_of_purchase'] == 2].groupby(['date_of_purchase'])['date_of_purchase'].count())

monthly_analysis_mar = pd.DataFrame(df[df['month_of_purchase'] == 3].groupby(['date_of_purchase'])['date_of_purchase'].count())

monthly_analysis_apr = pd.DataFrame(df[df['month_of_purchase'] == 4].groupby(['date_of_purchase'])['date_of_purchase'].count())

monthly_analysis_may = pd.DataFrame(df[df['month_of_purchase'] == 5].groupby(['date_of_purchase'])['date_of_purchase'].count())

monthly_analysis_jun = pd.DataFrame(df[df['month_of_purchase'] == 6].groupby(['date_of_purchase'])['date_of_purchase'].count())

monthly_analysis_jul = pd.DataFrame(df[df['month_of_purchase'] == 7].groupby(['date_of_purchase'])['date_of_purchase'].count())

monthly_analysis_aug = pd.DataFrame(df[df['month_of_purchase'] == 8].groupby(['date_of_purchase'])['date_of_purchase'].count())

monthly_analysis_sep = pd.DataFrame(df[df['month_of_purchase'] == 9].groupby(['date_of_purchase'])['date_of_purchase'].count())

monthly_analysis_oct = pd.DataFrame(df[df['month_of_purchase'] == 10].groupby(['date_of_purchase'])['date_of_purchase'].count())

monthly_analysis_nov = pd.DataFrame(df[df['month_of_purchase'] == 11].groupby(['date_of_purchase'])['date_of_purchase'].count())

monthly_analysis_dec = pd.DataFrame(df[df['month_of_purchase'] == 12].groupby(['date_of_purchase'])['date_of_purchase'].count())
fig = go.Figure()



fig.add_trace(go.Scatter(x=monthly_analysis_jan.index, y=monthly_analysis_jan['date_of_purchase'], name ='January',

                         line=dict(color='blue', width=2)))



fig.add_trace(go.Scatter(x=monthly_analysis_feb.index, y=monthly_analysis_feb['date_of_purchase'], name ='February',

                         line=dict(color='yellow', width=2)))



fig.add_trace(go.Scatter(x=monthly_analysis_mar.index, y=monthly_analysis_mar['date_of_purchase'], name ='March',

                         line=dict(color='green', width=2)))



fig.add_trace(go.Scatter(x=monthly_analysis_apr.index, y=monthly_analysis_apr['date_of_purchase'], name ='April',

                         line=dict(color='red', width=2)))



fig.add_trace(go.Scatter(x=monthly_analysis_may.index, y=monthly_analysis_may['date_of_purchase'], name ='May',

                         line=dict(color='grey', width=2)))



fig.add_trace(go.Scatter(x=monthly_analysis_jun.index, y=monthly_analysis_jun['date_of_purchase'], name ='June',

                         line=dict(color='black', width=2)))



fig.add_trace(go.Scatter(x=monthly_analysis_jul.index, y=monthly_analysis_jul['date_of_purchase'], name ='July',

                         line=dict(color='purple', width=2)))



fig.add_trace(go.Scatter(x=monthly_analysis_aug.index, y=monthly_analysis_aug['date_of_purchase'], name ='August', 

                         line=dict(color='violet', width=2)))



fig.add_trace(go.Scatter(x=monthly_analysis_sep.index, y=monthly_analysis_sep['date_of_purchase'], name ='September', 

                         line=dict(color='royalblue', width=2)))



fig.add_trace(go.Scatter(x=monthly_analysis_oct.index, y=monthly_analysis_oct['date_of_purchase'], name ='October', 

                         line=dict(color='darkblue', width=2)))



fig.add_trace(go.Scatter(x=monthly_analysis_nov.index, y=monthly_analysis_nov['date_of_purchase'], name ='November', 

                         line=dict(color='darkred', width=2)))



fig.add_trace(go.Scatter(x=monthly_analysis_dec.index, y=monthly_analysis_dec['date_of_purchase'], name ='December', 

                         line=dict(color='darkgreen', width=2)))



fig.update_layout(title='Monthly Analysis (Double click on name in legend)',xaxis_title="Date",yaxis_title="Number of Orders")



fig.show()
time_purchase= df.groupby(['time_of_purchase']).agg({'time_of_purchase':'count'}).rename(columns={'time_of_purchase':'Count'})
fig = go.Figure(data=go.Scatter(x=time_purchase.index,

                                y=time_purchase['Count'],

                                mode='markers',

                                marker=dict(

                                size=time_purchase['Count']*0.008))) # hover text goes here

fig.update_layout(title='Purchase Times',xaxis_title="Hour",yaxis_title="Number of Orders")

fig.show()
popular_cities_seller = pd.DataFrame(df['seller_city'].value_counts())

popular_cities_customer = pd.DataFrame(df['customer_city'].value_counts())
fig = go.Figure(go.Bar(y=popular_cities_seller.head(20).index, x=popular_cities_seller['seller_city'].head(20), 

                      orientation="h"))



fig.update_layout(title_text='Top Cities For Sellers',xaxis_title="No. of Orders")

fig.update_xaxes(type="log")

fig.show()
fig = go.Figure(go.Bar(y=popular_cities_customer.head(20).index, x=popular_cities_customer['customer_city'].head(20), 

                      orientation="h", marker=dict(color="green")))



fig.update_layout(title_text='Top Cities For Customers',xaxis_title="No. of Orders")

fig.update_xaxes(type="log")

fig.show()
df.head(1).T
#Most popular categories

popular_categories = pd.DataFrame(df.groupby(['product_category_name'])['order_id'].count().sort_values(ascending = False))
fig = go.Figure(go.Bar(

    x=popular_categories.head(10).index,y=popular_categories['order_id'].head(10),

    marker={'color': popular_categories['order_id'], 

    'colorscale': 'Viridis'},  

    text=popular_categories['order_id'],

    textposition = "outside",

))

fig.update_layout(title_text='Most popular categories',xaxis_title="Category Name",yaxis_title="No. of orders")

fig.show()
popular_cat_cities = pd.DataFrame(df.groupby(['customer_city','product_category_name'])['order_id'].count())
df_credit_card = df[df['payment_type'] == 'credit_card']

df_boleto = df[df['payment_type'] == 'boleto']

df_voucher = df[df['payment_type'] == 'voucher']

df_debit_card = df[df['payment_type'] == 'debit_card']
print('Distribution of payment menthods: \n')

print(df['payment_type'].value_counts())
print('Installment information for credit card transactions: \n')

print(df_credit_card['payment_installments'].describe())
print('Credit Card Payment Value\n')

print(df_credit_card['payment_value'].describe())
print('Boleto Payment Value\n')

print(df_boleto['payment_value'].describe())
print('Voucher Payment Value\n')

print(df_voucher['payment_value'].describe())
print('Debit Card Payment Value\n')

print(df_debit_card['payment_value'].describe())
plt.figure(figsize=(16,9))

sns.heatmap(df.corr())