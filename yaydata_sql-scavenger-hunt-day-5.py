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
# Examine the datasets.
#github.table_schema("sample_commits")
github.head("sample_commits")

#github.table_schema("sample_files")
github.head("sample_files")
# Build the query to answer the question.
# We will need the commits 
python_commits_query =  """ SELECT COUNT(DISTINCT c.commit) as commits, f.repo_name as repo
                                --select destinct commits to prevent double counting
                                --   which can occur because 1 commit effects multiple 
                                --   file in a repo
                            FROM `bigquery-public-data.github_repos.sample_commits` as c
                            INNER JOIN `bigquery-public-data.github_repos.sample_files` as f
                                ON f.repo_name = c.repo_name
                            WHERE f.path LIKE '%.py'
                            GROUP BY repo
                            ORDER BY commits DESC
                        """

# Estimate query size in GB.
#github.estimate_query_size(python_commits_query)

# Safely run the query.
python_commits = github.query_to_pandas_safe(python_commits_query, max_gb_scanned=6)
# Find the dimensions of the results.
import numpy as np
python_commits.shape
# Show the number of commits by repo.
print(python_commits)