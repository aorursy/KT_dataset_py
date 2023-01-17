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
transactions_per_month.head(5)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
# Your code goes here :)

#Question1: How many Bitcoin transactions were made each day in 2017?

MY_QUERY =  """    WITH time AS
                             (
                                 SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                                        transaction_id
                                 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                             )
                             SELECT EXTRACT(DAY FROM trans_time) AS day,
                                    EXTRACT(MONTH FROM trans_time) AS month,
                                    EXTRACT(YEAR FROM trans_time) AS year,
                                    COUNT(transaction_id) AS no_of_transaction
                             FROM time
                             GROUP BY day, month, year
                             HAVING year = 2017
                             ORDER BY month, day
                     """

#estimate query size
bitcoin_blockchain.estimate_query_size(MY_QUERY)
trans_in_year = bitcoin_blockchain.query_to_pandas_safe(MY_QUERY, max_gb_scanned=21)
trans_in_year
#Question2: How many transactions are associated with each merkle root?

MY_QUERY2 = """ SELECT COUNT(transaction_id) AS count, merkle_root
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                GROUP BY merkle_root
                ORDER BY count DESC

"""

bitcoin_blockchain.estimate_query_size(MY_QUERY2)
merkletrans_per_root = bitcoin_blockchain.query_to_pandas_safe(MY_QUERY2, max_gb_scanned=40)
merkletrans_per_root