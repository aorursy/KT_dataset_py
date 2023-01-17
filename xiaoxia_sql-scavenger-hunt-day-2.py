# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
print ("tables", hacker_news.list_tables())
# print the first couple rows of the "comments" table
hacker_news.head("comments")
# query to pass to 
query = """SELECT parent, COUNT(id) as number_of_comments
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)
print ("number of news with more then 10 comments: ", popular_stories.size)
popular_stories.head()
# Your code goes here :)

# import the package with helper functions
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                      dataset_name = "hacker_news")

# print the first handful of rows of the full table
hacker_news.head("full")
# How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
stories_by_id_query = """
            SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            HAVING count(id) > 0
            """

stories_by_id = hacker_news.query_to_pandas_safe(stories_by_id_query)
print("Number of stories of each type :", stories_by_id.size)
stories_by_id.head(10)