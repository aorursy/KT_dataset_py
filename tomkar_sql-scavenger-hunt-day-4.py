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
                EXTRACT(DAY FROM trans_time) AS day
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time)  = 2017
            GROUP BY month, day 
            ORDER BY month, day
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)


transactions_per_day.head()
# import plotting library
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (18,6)
# plot monthly bitcoin transactions
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations in 2017")

query = """
               WITH merkels AS (
                  SELECT merkle_root,
                         transaction_id
                  FROM `bigquery-public-data.bitcoin_blockchain.transactions`
               )
               SELECT merkle_root,
                    COUNT(DISTINCT transaction_id) AS transacts
                FROM merkels
                GROUP BY merkle_root
        """
# note that max_gb_scanned is set to 21, rather than 1
#block_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
block_per_merkle = bitcoin_blockchain.query_to_pandas(query)
#block_per_merkle.get_values('transacts').sort(transacts,ascendent=False)
block_per_merkle.transacts.sort_values(ascending=False)