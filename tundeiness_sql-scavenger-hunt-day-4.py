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
# Import the package with the helper function
import bq_helper


# Create a helper object for this dataset
bitcoin_data = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="bitcoin_blockchain")
#View datasets
bitcoin_data.head("blocks")
#View datasets
bitcoin_data.head("transactions")
bquery = """ WITH trans_time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS time_conv,
                    transactions
                FROM `bigquery-public-data.bitcoin_blockchain.blocks`
            )
            SELECT COUNT(transactions) AS trans_count,
                EXTRACT(DAY FROM time_conv) AS day
            FROM trans_time
            GROUP BY day, day 
            ORDER BY day, day
        """
# Convert to pandas dataframe
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(bquery, max_gb_scanned=21)
# import plotting library
import matplotlib.pyplot as plt

# plot daily bitcoin transactions
plt.plot(transactions_per_day)
plt.title("Daily Bitcoin Transcations")
transactions_per_day.head()
merkle_root = """
                SELECT COUNT(transaction_id) AS trans_count, merkle_root AS merkle
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                GROUP BY merkle_root
            """
# Convert to pandas dataframe
transactions_per_merkle = bitcoin_blockchain.query_to_pandas_safe(merkle_root, max_gb_scanned=37)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_merkle)
plt.title("Transcations per Merkle")

transactions_per_merkle.head()
trans_id = """
                SELECT COUNT(transaction_id) AS trans_count, block_id AS block
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                GROUP BY block_id
                ORDER BY trans_count
            """
# Convert to pandas dataframe
transactions_per_block = bitcoin_blockchain.query_to_pandas_safe(trans_id, max_gb_scanned=37)
transactions_per_block.head()