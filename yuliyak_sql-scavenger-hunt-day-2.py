# import helper
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                                      dataset_name = "hacker_news")

# print first rows of "comments" table
hacker_news.head("comments")
# query to pass to
query = """SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """
# safely run the query
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
# Check out "full" table.
hacker_news.head("full")
# Create a query.
# Count "ID" table and group by "type".
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
# Run the query safely
stories = hacker_news.query_to_pandas_safe(query)
# Check the results
stories.head()
# Create a query
query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted
            HAVING deleted = True
        """
# Run query safely
deleted = hacker_news.query_to_pandas_safe(query)
# Show the results
deleted.head()
# Query to pass
query = """SELECT type, MAX(score)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
# Run the query and see results
score = hacker_news.query_to_pandas_safe(query)

score.head()