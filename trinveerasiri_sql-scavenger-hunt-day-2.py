# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
# query to pass to 
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
story_by_type = hacker_news.query_to_pandas_safe(query)
story_by_type.head()
# query to pass to 
query = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
        """
deleted_comments = hacker_news.query_to_pandas_safe(query)
deleted_comments.head()