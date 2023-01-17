import bq_helper
import matplotlib.pyplot as plt

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
bitcoin_blockchain.estimate_query_size(query)
# note that max_gb_scanned is set to 21, rather than 1
#transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)

#plt.plot(transactions_per_month.transactions)
#plt.title("Monthly Bitcoin Transcations")
query = """ WITH time AS
            (
                    SELECT TIMESTAMP_MILLIS(timestamp) AS timestamps,
                        transaction_id
                    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT EXTRACT(DAY FROM timestamps) AS day,
                EXTRACT(MONTH FROM timestamps) AS month,
                COUNT(transaction_id) AS num_transactions
            FROM time
            WHERE EXTRACT(YEAR FROM timestamps)=2017
            GROUP BY month, day
            ORDER BY month, day
        """
bitcoin_blockchain.estimate_query_size(query)
transactions_day=bitcoin_blockchain.query_to_pandas_safe(query,max_gb_scanned=21)
plt.plot(transactions_day.num_transactions)
query = """SELECT COUNT(transaction_id) AS num_transactions,
                merkle_root
           FROM `bigquery-public-data.bitcoin_blockchain.transactions`
           GROUP BY merkle_root
           ORDER BY num_transactions
        """
bitcoin_blockchain.estimate_query_size(query)
transaction_merkle=bitcoin_blockchain.query_to_pandas_safe(query,max_gb_scanned=37)
transaction_merkle.head()