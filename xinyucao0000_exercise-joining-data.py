# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
github.list_tables()
github.head("sample_commits")
github.head("sample_files")
# Your code here
query = """
        SELECT sf.repo_name, COUNT(sc.commit) as number_of_commits
        FROM `bigquery-public-data.github_repos.sample_commits` as sc
        INNER JOIN `bigquery-public-data.github_repos.sample_files` as sf ON sc.repo_name = sf.repo_name
        WHERE path LIKE '%.py'
        GROUP BY sf.repo_name
        ORDER BY number_of_commits
        """
python_commit = github.query_to_pandas(query)
python_commit