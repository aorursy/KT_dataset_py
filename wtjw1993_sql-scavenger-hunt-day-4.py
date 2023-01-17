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
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
query1 = """ WITH time AS
             (
                 SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                     transaction_id
                 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                 WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
             )
             SELECT EXTRACT(DAYOFYEAR FROM trans_time) as day,
                    COUNT(transaction_id) as transactions
             FROM time
             GROUP BY day
             ORDER BY day
         """
daily_trans = bitcoin_blockchain.query_to_pandas(query1)
plt.plot(daily_trans.transactions)
plt.title('Number of Daily Transactions in 2017')
plt.xlabel('Day of Year')
plt.ylabel('Number of transactions')
plt.show()
query2 = """ SELECT merkle_root, 
                    COUNT(transaction_id) as transactions
             FROM `bigquery-public-data.bitcoin_blockchain.transactions`
             GROUP BY merkle_root
             ORDER BY transactions DESC
         """
merkle_id_trans = bitcoin_blockchain.query_to_pandas(query2)
merkle_id_trans.sort_values('transactions').head()