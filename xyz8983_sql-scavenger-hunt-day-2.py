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
hacker_news.list_tables()
hacker_news.head("full")



query_count_type = """SELECT type, COUNT(id)
                      FROM `bigquery-public-data.hacker_news.full`
                      GROUP BY type
                   """
count_of_each_type = hacker_news.query_to_pandas_safe(query_count_type)
count_of_each_type.head()
query_count_deleted = """SELECT COUNT(id)
                         FROM `bigquery-public-data.hacker_news.full`
                         WHERE deleted is TRUE
                        """
count_of_deleted = hacker_news.query_to_pandas_safe(query_count_deleted)
count_of_deleted.head()