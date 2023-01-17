# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
github.head("licenses")
github.head("files", selected_columns = ["repo_name", "id"],
    num_rows = 10)
#How many files are covered by each license? (not the right way to do it!!!)
MY_QUERY = """ SELECT COUNT(f.id) AS count, l.license AS license
                FROM `bigquery-public-data.github_repos.files` AS f 
                INNER JOIN `bigquery-public-data.github_repos.licenses` AS l
                ON f.repo_name = l.repo_name
                GROUP BY l.license
                ORDER BY count

"""

github.estimate_query_size(MY_QUERY)
github.query_to_pandas(MY_QUERY)
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
#checking head of both files
github.head("sample_commits")
#checking the head
github.head("sample_files")
# Your code goes here :)

# question: How many commits (recorded in the "sample_commits" table) 
# have been made in repos written in the Python programming language? 
#(I'm looking for the number of commits per repo for all the repos written in Python

MY_QUERY1 = """ SELECT COUNT(sc.commit) AS count, sf.repo_name
                FROM `bigquery-public-data.github_repos.sample_files` AS sf
                INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS sc 
                ON sf.repo_name = sc.repo_name
                WHERE sf.path LIKE '%.py'
                GROUP BY sf.repo_name
                ORDER BY count 

"""

github.estimate_query_size(MY_QUERY1)
commits_per_repo_python = github.query_to_pandas(MY_QUERY1)
commits_per_repo_python
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10,20))
plt.bar(commits_per_repo_python.repo_name, commits_per_repo_python.count, align='center', alpha=0.5)
#easiest way to plot. USE pandas

commits_per_repo_python.plot('repo_name', 'count', kind='bar', color='r')