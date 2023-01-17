import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper as bqh #the big query helper makes working with big query easier
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
#handy utility function
def GetDescription(dataset):
    #check out the data structure
    for table in dataset.list_tables():
        print('_________\n',table,'\n________')
        for field in dataset.table_schema(table):
            print('\t',field)
        #print(dataset.head(table))
#get the dataset
project="bigquery-public-data"
dataset="bitcoin_blockchain"
bit_blocks = bqh.BigQueryHelper(active_project=project, dataset_name=dataset)
#GetDescription(bit_blocks)
# build the query containing the data we need to answer the first question
txquery = """ WITH filteredtx AS
            (SELECT 
                EXTRACT(YEAR FROM CAST(TIMESTAMP_MILLIS(timestamp) AS DATETIME)) as transactionyear,
                EXTRACT(DAY FROM CAST(TIMESTAMP_MILLIS(timestamp) AS DATETIME)) as transactionday,
                EXTRACT(MONTH FROM CAST(TIMESTAMP_MILLIS(timestamp) AS DATETIME)) as transactionmonth,
                CAST(TIMESTAMP_MILLIS(timestamp) AS DATE) as transactiondate,
                transaction_id
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            WHERE EXTRACT(YEAR FROM CAST(TIMESTAMP_MILLIS(timestamp) AS DATETIME)) = 2017
            ) 
            SELECT 
                transactionyear,
                transactionday,
                transactionmonth,
                transactiondate,
                count(transaction_id) as transactioncount
            FROM filteredtx
            GROUP BY transactionyear,transactionday,transactionmonth,transactiondate"""

# check how big the tx query will be
print('tx query size',bit_blocks.estimate_query_size(txquery))
#once we know the tx query is small enough, run it
tx_df=bit_blocks.query_to_pandas_safe(txquery,max_gb_scanned=21)
#confirm that we have a result 
#note: it may be necessary to up the max_gb_scanned if no results are found
if tx_df is None:
    print("no data found")
else:
    #order matters when plotting, so let's sort our dataset
    tx_df.sort_values(by="transactiondate",inplace=True)
    #the overall mean is generally interesting when
    #answering this type of question, so we'll calculate it
    meantxperday=round(tx_df['transactioncount'].mean(),2)
    date=tx_df["transactiondate"] #axis values

    #set up the plot
    fig, ax = plt.subplots()
    plt.title("2017 Bitcoin Transactions Per Day\n(mean "+str(meantxperday)+")")
    ax.plot(date, tx_df.transactioncount)

    # format the date ticks
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%m'))

    #set the min and max plot years
    datemin = datetime.date(date.min().year, 1, 1)
    datemax = datetime.date(date.max().year+1,1,1)
    ax.set_xlim(datemin, datemax)

    #add a plot line showing the mean
    plt.axhline(y=meantxperday, color='g', linestyle='-')
    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()
    plt.show()
# build the query containing the data we need to answer the second question
#first, we get a list of merkle roots and the number of blocks each one has
#then, we get a list of transaction counts and how many merkle roots have that transaction count
mbquery = """ WITH tx_counts AS
            (SELECT 
                merkle_root,
                COUNT(transaction_id) as txcount
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ) 
            SELECT 
                count(merkle_root) as merkle_count,
                txcount
            FROM tx_counts
            GROUP BY txcount"""
# check how big the tx query will be
print('mb query size',bit_blocks.estimate_query_size(mbquery))
#once we know the tx query is small enough, run it
mb_df=bit_blocks.query_to_pandas_safe(mbquery,max_gb_scanned=37)
#confirm that we have a result 
#note: it may be necessary to up the max_gb_scanned if no results are found
if mb_df is None:
    print("no data found")
else:
    mb_df.sort_values(by="merkle_count",inplace=True)
    plt.scatter(y=mb_df["merkle_count"],x=mb_df["txcount"])
    plt.xlabel("number of transactions")
    plt.ylabel("merkle roots")
    plt.title("Transactions per Merkle Root")
