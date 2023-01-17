# boilerplate from the original notebook
import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# for visualization
import matplotlib.pyplot as plt
import seaborn as sns
query = """
  SELECT COUNT(DISTINCT id) AS ids, type
  FROM `bigquery-public-data.hacker_news.full`
  GROUP BY type
"""
hacker_news.estimate_query_size(query)
ids_per_type = hacker_news.query_to_pandas_safe(query)

plt.subplots(figsize=(11.7, 8.27))
sns.barplot(x="type", y="ids", data=ids_per_type)
plt.title("How many IDs belong to each type?")
query = """
  SELECT COUNT(DISTINCT id) AS comments
  FROM `bigquery-public-data.hacker_news.full`
  WHERE type = 'comment' AND deleted = TRUE
"""
hacker_news.estimate_query_size(query)
deleted_comments = hacker_news.query_to_pandas_safe(query)
deleted_comments.comments
query = """
  SELECT COUNT(DISTINCT id) AS stories
  FROM `bigquery-public-data.hacker_news.full`
  WHERE type = 'story'
"""
hacker_news.estimate_query_size(query)
story_count = hacker_news.query_to_pandas_safe(query)
story_count.stories
query = """
  SELECT 
    AVG(score) AS mean,
    
    # BigQuery only supports approx quantiles. 
    # I wonder how difficult this problem is for parallelism...
    APPROX_QUANTILES(score, 4) AS quartiles
  FROM `bigquery-public-data.hacker_news.full`
  WHERE type = 'story'
"""
hacker_news.estimate_query_size(query)
score_statistics = hacker_news.query_to_pandas_safe(query)
score_statistics['mean']
score_statistics['quartiles']