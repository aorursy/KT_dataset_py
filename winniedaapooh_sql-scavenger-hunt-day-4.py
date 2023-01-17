# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, month 
            ORDER BY year, month
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_per_month['time-mark'] = [ str(transactions_per_month.month[i]) + '-' + str(transactions_per_month.year[i]) for i,year in enumerate(transactions_per_month.year)]
transactions_per_month
fig, ax = plt.subplots(figsize=(12,8))
#ax.yaxis.set_major_formatter(formatter)
import numpy as np
import pandas as pd
x = np.arange(110)
date_range = pd.date_range(start='20090101',end='20180201',freq='MS')
print(len(date_range))
plt.plot(date_range, transactions_per_month.transactions)
plt.title("Number of Transactions Each Month")
plt.show()
transactions_per_month['MoM_Pct'] = transactions_per_month.transactions.pct_change(periods=12).clip(upper=5)
## Clip at 500% for better visualization
date_range = pd.date_range(start='20090101',end='20180201',freq='MS')
print(len(date_range))
plt.plot(date_range, transactions_per_month.MoM_Pct)
plt.title("Year over Year Percentage Change from 2009-2018")
plt.show()
# Your code goes here :)
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                        transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM trans_time) AS day,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            where EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY year, month , day
            ORDER BY year, month , day
        """
transactions_daily_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
y_mean = transactions_daily_2017.transactions.mean()

## Clip at 500% for better visualization
date_range = pd.date_range(start='20170101',end='20171231',freq='D')
plt.plot(date_range, transactions_daily_2017.transactions)
y_mean = [transactions_daily_2017.transactions.mean()]*len(date_range)
mean_line = plt.plot(date_range,y_mean, label='Mean',color='r', linestyle='--')

plt.title("Daily Transactions in 2017")
plt.show()
query = """ WITH merkle_df AS 
            (
                SELECT count(transaction_id) AS COUNT,
                        merkle_root AS MERKEL
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                group by merkle_root
            )
            select * from merkle_df
            order by COUNT DESC
            """
merkle_count_df = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=37)
merkle_count_df.head(n=10)
len(merkle_count_df)