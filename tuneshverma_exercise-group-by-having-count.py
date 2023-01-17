# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("full")
hacker_news.head("comments")
query="""SELECT type, COUNT(id)
         FROM `bigquery-public-data.hacker_news.full`
         GROUP BY type"""
         
hacker_news_type=hacker_news.query_to_pandas_safe(query)
hacker_news_type.head()
query_2="""SELECT deleted, COUNT(id)
         FROM `bigquery-public-data.hacker_news.full`
         GROUP BY deleted
         HAVING deleted=True"""
hacker_news_deleted=hacker_news.query_to_pandas_safe(query_2)
hacker_news_deleted
# Your Code Here