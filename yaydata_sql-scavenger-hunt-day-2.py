# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
# query to pass to 
query = """SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
# Your code goes here :)
# look at the contents of the full table.
hacker_news.table_schema('full')
# Write and run a query to answer question 1.
query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
         """
# Estimate query size in GB.
#hacker_news.estimate_query_size(query1)

popular_stories = hacker_news.query_to_pandas_safe(query1, max_gb_scanned=0.5)
popular_stories.to_csv("popular_stories.csv")
# Write and run a query to answer question 2. 
query2 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted
            HAVING deleted=TRUE
         """
# Estimate query size in GB.
hacker_news.estimate_query_size(query2)

deleted_comments = hacker_news.query_to_pandas_safe(query2)
deleted_comments.to_csv("deleted_comments.csv")
# Write and runa query to answer the extra credit. Use a COUNTIF instead of COUNT
#   and HAVING statements.
query3 = """SELECT COUNTIF(deleted=TRUE)
            FROM `bigquery-public-data.hacker_news.full`
         """
# Estimate query size in GB.
#hacker_news.estimate_query_size(query3)

deleted_comment2 = hacker_news.query_to_pandas_safe(query3)
deleted_comment2.to_csv("deleted_comment2.csv")