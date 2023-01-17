# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.list_tables()
bitcoin_blockchain.head('transactions')
bitcoin_blockchain.head('blocks')
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
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=23)
transactions_per_month.head(15)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
# Your code goes here :)
query1 = """ WITH time AS 
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
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY year, month, day
            ORDER BY year, month, day
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day_in_2017 = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=23)

transactions_per_day_in_2017.head()
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_day_in_2017.transactions)
plt.title("Daily Bitcoin Transcations in 2017")
query2 = """ WITH merkle AS 
            (
                SELECT merkle_root AS merkle_root,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions, merkle_root
            FROM merkle
            
            GROUP BY merkle_root
            
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=41)
transactions_per_merkle_root.count()