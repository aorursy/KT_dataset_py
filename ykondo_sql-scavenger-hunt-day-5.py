# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
query = ("""
        SELECT sf.repo_name, COUNT(sc.commit) AS number_of_commits
        FROM `bigquery-public-data.github_repos.sample_commits` as sc
        INNER JOIN `bigquery-public-data.github_repos.sample_files` as sf
            ON sf.repo_name = sc.repo_name
        WHERE sf.path LIKE '%.py'
        GROUP BY sf.repo_name
        ORDER BY number_of_commits DESC
        """)

commit_count_by_repo = github.query_to_pandas_safe(query, max_gb_scanned=6)
commit_count_by_repo