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
# counting the number of sotries by type
query1 = """SELECT type, COUNT(id) as count
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
         """
stories_by_type = hacker_news.query_to_pandas_safe(query1)
stories_by_type
# deleted comments
query2 = """SELECT COUNT(id) as count
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
         """
deleted_comments = hacker_news.query_to_pandas_safe(query2)
deleted_comments
# smallest and largest id number for each type of story
query3 = """SELECT type, MIN(id) as smallest_id, MAX(id) as largest_id
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
         """
id_range = hacker_news.query_to_pandas_safe(query3)
id_range