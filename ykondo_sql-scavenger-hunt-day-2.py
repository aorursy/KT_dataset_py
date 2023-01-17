# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "full" table
hacker_news.head("full")
# query to pass to 
query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
count_stories = hacker_news.query_to_pandas_safe(query1)
count_stories
# query to pass to 
query2 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted
        """
count_deleted = hacker_news.query_to_pandas_safe(query2)
count_deleted
# query to pass to 
query3 = """SELECT type, AVG(score)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
averagescore_type = hacker_news.query_to_pandas_safe(query3)
averagescore_type