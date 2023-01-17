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

query_commits = """
                SELECT COUNT(sc.commit) AS commits_number, sc.repo_name
                FROM  `bigquery-public-data.github_repos.sample_commits` AS sc
                JOIN  `bigquery-public-data.github_repos.sample_files` AS  sf
                      ON  sc.repo_name = sf.repo_name
                WHERE sf.path LIKE '%.py'
                GROUP BY sc.repo_name
                ORDER BY  commits_number DESC
                
                """
number_of_commits =  github.query_to_pandas_safe(query_commits, max_gb_scanned=6)
import matplotlib.pyplot as plt

plt.plot(number_of_commits.commits_number)
plt.title("Number of commits of repos written in Python")
number_of_commits.head()