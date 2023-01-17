# import package with helper functions 
import bq_helper
import matplotlib.pyplot as plt

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
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
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)
transactions_per_day.head()
# plot monthly bitcoin transactions
f, ax = plt.subplots(figsize=(14, 9)) 
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations in 2017")
query2 = """ SELECT COUNT(block_id) AS block_count, merkle_root
                FROM `bigquery-public-data.bitcoin_blockchain.blocks`
            GROUP BY merkle_root
            ORDER BY block_count
        """

# note that max_gb_scanned is set to 21, rather than 1
blocks_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=21)
blocks_per_merkle_root.head()