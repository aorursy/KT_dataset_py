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
#build query with CTE to get transactions for each day
query = """
        WITH time AS
        (
            SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                   transaction_id
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        )
        SELECT COUNT(transaction_id),
               EXTRACT(DAY from trans_time) AS trans_day
        FROM time
        GROUP BY trans_day
        ORDER BY COUNT(transaction_id) DESC
        """
#run query with adjustment to scan size
transactions_by_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
#build query with CTE to get transactions associated with each merkle root
query = """
        WITH roots AS
        (
            SELECT transaction_id,
                   merkle_root AS root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        )
        SELECT COUNT(transaction_id),
               root
        FROM roots
        GROUP BY root
        ORDER BY COUNT(transaction_id) DESC
        """
#run query with adjustment to scan size
transactions_by_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=38)