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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

bitcoin_blockchain.list_tables()
bitcoin_blockchain.head("transactions")

# 1. How many Bitcoin transactions were made each day in 2017?
query1 = """
            WITH time AS (
            SELECT transaction_id AS id, 
                   TIMESTAMP_MILLIS(timestamp) AS trans_time
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT EXTRACT(DAY FROM trans_time) AS day, 
                   EXTRACT(MONTH FROM trans_time) AS month,
                   EXTRACT(YEAR FROM trans_time) AS year,
                   COUNT(id) AS cnt
            FROM time
            GROUP BY year, month, day
            HAVING year = 2017
            ORDER BY cnt DESC
         """

transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)
transactions_per_day.head()

# 2. How many transactions are associated with each merkle root?
query2 = """
         SELECT merkle_root, COUNT(transaction_id) AS cnt
         FROM `bigquery-public-data.bitcoin_blockchain.transactions`
         GROUP BY merkle_root
         ORDER BY cnt DESC
         """
transactions_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=37)
transactions_per_merkle.head()