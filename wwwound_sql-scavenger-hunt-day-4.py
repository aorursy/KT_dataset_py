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
# Your code goes here :)
query = """
    WITH transactions AS (
      SELECT  
        TIMESTAMP_MILLIS(`TIMESTAMP`) AS ts,
        transaction_id
      FROM `bigquery-public-data.bitcoin_blockchain.transactions` 
    )

    SELECT
      DATE(ts) as `date`,
      COUNT(transaction_id) as transactions
    FROM transactions
    WHERE EXTRACT(YEAR FROM ts) = 2017
    GROUP BY `date`
    ORDER BY `date`
"""
transactions_per_day_in_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
plt.plot(transactions_per_day_in_2017.transactions)
plt.title("Bitcoin Transactions by day in 2017")
plt.show()

query = """
    WITH transactions AS (
      SELECT  
        merkle_root,
        COUNT(transaction_id) as transactions_by_merkle_root
      FROM `bigquery-public-data.bitcoin_blockchain.transactions` 
      GROUP BY merkle_root
    )

    SELECT
      transactions_by_merkle_root,
      COUNT(merkle_root) as `count`
    FROM transactions
    GROUP BY transactions_by_merkle_root
    ORDER BY `count` DESC
"""
transactions_count_by_merkle_root_count = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=37)
print(transactions_count_by_merkle_root_count.head(n=20))