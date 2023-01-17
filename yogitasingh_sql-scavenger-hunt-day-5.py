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
#number of commits per repo for all the repos written in Python
query = ("""SELECT sf.path, COUNT(commit) as number_of_commits
FROM `bigquery-public-data.github_repos.sample_commits` AS sc
INNER JOIN `bigquery-public-data.github_repos.sample_files` as sf
ON sc.repo_name = sf.repo_name
WHERE path LIKE '%.py'
GROUP BY sf.path 
 """ )
pcommits_per_repo = github.query_to_pandas_safe(query, max_gb_scanned=6)
pcommits_per_repo.head()
#number of commits per repo for all the repos written in Python
query = ("""SELECT sf.path, COUNT(commit) as number_of_commits
FROM `bigquery-public-data.github_repos.sample_commits` AS sc
INNER JOIN `bigquery-public-data.github_repos.sample_files` as sf
ON sc.repo_name = sf.repo_name
WHERE path NOT LIKE '%.py'
GROUP BY sf.path 
 """ )
commits_per_repo = github.query_to_pandas_safe(query, max_gb_scanned=14)

commits_per_repo.head()