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
query_number_stories = """SELECT  type, COUNT(id)
                            FROM `bigquery-public-data.hacker_news.full`
                            GROUP BY type
                        """
number_stories = hacker_news.query_to_pandas_safe(query_number_stories)
print(number_stories)

query_deleted = """SELECT COUNT(id) as number_comment_deleted, deleted
                    FROM `bigquery-public-data.hacker_news.comments`
                    GROUP BY deleted
                    HAVING deleted = True
                    
                """
number_deleted = hacker_news.query_to_pandas_safe(query_deleted)
print(number_deleted)
query_comment_deleted = """SELECT COUNT(deleted) as deleted_comment
                            FROM `bigquery-public-data.hacker_news.comments`
                        """
number_comment_deleted = hacker_news.query_to_pandas_safe(query_comment_deleted)
print(number_comment_deleted)