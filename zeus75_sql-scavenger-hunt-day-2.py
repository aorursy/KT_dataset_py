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

# import the package with helper functions
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                      dataset_name = "hacker_news")


# print the first handful of rows of the full table
hacker_news.head("full")
# generate our first query ("how many stores of each type")
type_query = """
            SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            HAVING count(id) > 0
            """
# generate a safe execution method
type_query = hacker_news.query_to_pandas_safe(type_query)
print(type_query.head())
# generate our second query (Count deleted comments)
deleted_query = """ SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = TRUE
            """
# generate a safe execution method
deleted_count = hacker_news.query_to_pandas_safe(deleted_query)
print(deleted_count.head())