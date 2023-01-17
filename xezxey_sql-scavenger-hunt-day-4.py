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
print(transactions_per_month)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
query_number_bitcoin_transcations = """WITH transaction AS
                                        (
                                            SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                                                transaction_id
                                            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                                        )
                                        SELECT COUNT(transaction_id) AS number_transactions, EXTRACT(DAYOFYEAR FROM trans_time) AS day
                                        FROM transaction
                                        GROUP BY day
                                        ORDER BY COUNT(transaction_id)
                                    """

number_bitcoin_transaction = bitcoin_blockchain.query_to_pandas_safe(query_number_bitcoin_transcations, max_gb_scanned = 21)
print(number_bitcoin_transaction)

plt.plot(number_bitcoin_transaction.number_transactions)
plt.title("Number of Bitcoin Transaction")
query_transaction_associated_merkle_root = """SELECT merkle_root, COUNT(transaction_id) AS number_of_transactions
                                                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                                                GROUP BY merkle_root
                                            """

transaction_associated_merkle_root = bitcoin_blockchain.query_to_pandas(query_transaction_associated_merkle_root)
print(transaction_associated_merkle_root)
plt.plot(transaction_associated_merkle_root.number_of_transactions)
plt.title("Associated Transaction")