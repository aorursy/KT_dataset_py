# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
github.list_tables()
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
file_count_by_license.head()
# print out all the returned results
print(file_count_by_license)
github.head('sample_files')
github.table_schema('sample_files')
github.head('sample_commits')
github.table_schema('sample_commits')
# Your code goes here :)

# first intent, we'll see this is not the correct answer by further analysis.
query1 = ("""
        SELECT 
        sf.repo_name, 
        COUNT(DISTINCT sc.commit) AS number_of_files
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc
            ON sf.repo_name = sc.repo_name
        WHERE path LIKE '%.py'
        GROUP BY sf.repo_name
        ORDER BY number_of_files DESC
        """)
github.estimate_query_size(query1)
commits_in_python = github.query_to_pandas_safe(query1, max_gb_scanned=6)
commits_in_python
import matplotlib as plt
commits_in_python.plot()