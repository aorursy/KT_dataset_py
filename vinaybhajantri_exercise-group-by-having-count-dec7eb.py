# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("full")
#hacker_news.list_tables()

import bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

query = """ SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
stories = hacker_news.query_to_pandas_safe(query)
print(stories.head())
import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
query = """ SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted
            HAVING deleted = True
        """

deleted_comments = hacker_news.query_to_pandas_safe(query)
print(deleted_comments.head())
import bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

query = """ SELECT type, COUNTIF(deleted = True)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
stories = hacker_news.query_to_pandas_safe(query)
print(stories.head())