# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
# You can use two dashes (--) to add comments in SQL
query = ("""
        -- Select all the columns we want in our joined table
        SELECT L.license, COUNT(sf.path) AS number_of_files
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.licenses` as L 
            ON sf.repo_name = L.repo_name -- what columns should we join on?
        GROUP BY L.license
        ORDER BY number_of_files DESC
        """)

file_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned=6)
# print out all the returned results
print(file_count_by_license)
# Double check the CTE table
# commits_query = ("""
#             SELECT files.repo_name
#             FROM `bigquery-public-data.github_repos.sample_files` AS files
#             WHERE files.path LIKE '%.py'
#             GROUP BY files.repo_name
#         """)

commits_query = ("""
        WITH python_repo AS (
            SELECT files.repo_name
            FROM `bigquery-public-data.github_repos.sample_files` AS files
            WHERE files.path LIKE '%.py'
            GROUP BY files.repo_name
        )
        SELECT pr.repo_name,
            COUNT(commits.commit) AS number_of_commits
        FROM python_repo as pr
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as commits 
            ON pr.repo_name = commits.repo_name
        GROUP BY pr.repo_name
        ORDER BY number_of_commits DESC
        """)

commits_per_repo = github.query_to_pandas_safe(commits_query, max_gb_scanned=6)
commits_per_repo