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
import bq_helper

bitcoin_blockchain =bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                             dataset_name="bitcoin_blockchain")
query01v2 = \
"""

    WITH day_transaction1 AS
    (
        WITH day_transaction AS
        (
            SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        )
        SELECT COUNT(transaction_id) AS num_of_transactions,
               EXTRACT(DAY FROM trans_time) AS day,
               EXTRACT(MONTH FROM trans_time) AS month,
               EXTRACT(YEAR FROM trans_time) AS year
        FROM day_transaction
        GROUP BY year, month, day
        HAVING year = 2017
        ORDER BY month, day
    )
    SELECT num_of_transactions,
           CONCAT(CAST(day AS STRING), ' ', CAST(month AS STRING)) AS day_month
    FROM day_transaction1
"""

bitcoin_blockchain.estimate_query_size(query01v2)
bitcoin_transactions_by_day_in_2017 = bitcoin_blockchain.query_to_pandas(query01v2)
bitcoin_transactions_by_day_in_2017
plt.bar(bitcoin_transactions_by_day_in_2017['day_month'],
            bitcoin_transactions_by_day_in_2017['num_of_transactions'])
query02 = \
"""
    SELECT COUNT(transaction_id) AS num_of_transactions, merkle_root
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    GROUP BY merkle_root
    ORDER BY num_of_transactions DESC
"""
bitcoin_blockchain.estimate_query_size(query02)
num_of_transacts_by_merkle_root = bitcoin_blockchain.query_to_pandas(query02)
num_of_transacts_by_merkle_root
plt.hist(num_of_transacts_by_merkle_root['num_of_transactions'], bins=100)