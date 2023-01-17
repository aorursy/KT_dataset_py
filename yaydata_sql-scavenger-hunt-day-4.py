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
import numpy as np

# Examine the transactions dataset.
bitcoin_blockchain.head("transactions")
transactions_by_day_query = """WITH trans_times AS
                            (
                                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                                    transaction_id
                                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                            )
                            SELECT EXTRACT(DAY FROM trans_time) AS day, COUNT(transaction_id) AS transactions
                            FROM trans_times
                            WHERE EXTRACT(YEAR FROM trans_time) = 2017
                            GROUP BY day
                            ORDER BY day
                           """
# Estimate query size in GB.
#bitcoin_blockchain.estimate_query_size(transactions_by_day_query)

transactions_2017 = bitcoin_blockchain.query_to_pandas_safe(transactions_by_day_query, max_gb_scanned=21)
#transactions_2017.head()

import matplotlib.pyplot as plt
plt.plot(transactions_2017.transactions)
plt.title("Bitcoin Transactions 2017")
transactions_per_merkle_query = """ SELECT COUNT(transaction_id) AS transactions, merkle_root 
                                    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                                    GROUP BY merkle_root
                                """
# Estimate query size in GB.
#bitcoin_blockchain.estimate_query_size(transactions_per_merkle_query)

transactions_per_merkle = bitcoin_blockchain.query_to_pandas_safe(transactions_per_merkle_query, max_gb_scanned = 37)
# Examine the head.
transactions_per_merkle.head(n=10)
# Find the dimensions of the dataset.
transactions_per_merkle.shape