%matplotlib inline
import pandas as pd
train_df = pd.read_csv('../input/train.csv', parse_dates=['Date'])
date_store_indexed_train_df = train_df.sort_values(['Date', 'Store']).set_index(['Date', 'Store'])
num_df = date_store_indexed_train_df[['Sales', 'Customers', 'Open', 'Promo']]
pd.expanding_mean(num_df, 180).plot(subplots=True)
