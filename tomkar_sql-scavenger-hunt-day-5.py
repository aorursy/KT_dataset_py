# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
github.head("sample_commits")

github.head("sample_files")
# You can use two dashes (--) to add comments in SQL
query = ("""
        SELECT a.repo_name, COUNT(b.commit) AS commits_cnt
        FROM `bigquery-public-data.github_repos.sample_files` as a
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as b 
            ON a.repo_name = b.repo_name 
        WHERE path LIKE '%.py'
        GROUP BY a.repo_name
        ORDER BY commits_cnt DESC
        """)

commits_count_by_repo = github.query_to_pandas_safe(query, max_gb_scanned=6)

commits_count_by_repo