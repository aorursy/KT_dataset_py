# import package with helper functions 
import bq_helper
# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="bitcoin_blockchain")
bitcoin_blockchain.head('transactions')
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
            GROUP BY year, month, day 
            HAVING year = 2017
            ORDER BY year, month, day
        """
bitcoin_eachday = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
bitcoin_eachday
query = """ WITH merkle AS
        (
            SELECT transaction_id, merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions` 
        )
            SELECT COUNT(transaction_id) AS transactions, merkle_root
            FROM merkle
            GROUP BY merkle_root
        """
merkle_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=50)
merkle_root