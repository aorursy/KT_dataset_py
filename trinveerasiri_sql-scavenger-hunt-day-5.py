# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="github_repos")
github.head('sample_commits')
github.head('sample_files')
# You can use two dashes (--) to add comments in SQL
query = """WITH L AS 
        (
        SELECT repo_name
        FROM `bigquery-public-data.github_repos.sample_files`
        WHERE path LIKE '%.py'
        ) 
        SELECT R.repo_name, COUNT(commit) AS num_commits
        FROM `bigquery-public-data.github_repos.sample_commits` as R
        INNER JOIN L
        ON L.repo_name = R.repo_name
        GROUP BY R.repo_name
        ORDER BY num_commits DESC
        """
python_git = github.query_to_pandas_safe(query, max_gb_scanned=10)
python_git.head()