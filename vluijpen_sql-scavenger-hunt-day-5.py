# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
# You can use two dashes (--) to add comments in SQL
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")

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
# Your code goes here :)
#Question1
#How many commits (recorded in the "sample_commits" table) have been made in repos 
#written in the Python programming language? 
#(I'm looking for the number of commits per repo for all the repos written in Python.

query_commit = '''WITH python_repos AS (
    SELECT repo_name
    FROM `bigquery-public-data.github_repos.sample_files`
    WHERE path LIKE '%.py')
SELECT commits.repo_name, COUNT(commit) AS commit_count
FROM `bigquery-public-data.github_repos.sample_commits` AS commits
INNER JOIN python_repos
    ON  python_repos.repo_name = commits.repo_name
GROUP BY commits.repo_name
ORDER BY commit_count DESC

'''
github.estimate_query_size(query_commit)     #query size is approx 5GB
num_commits = github.query_to_pandas_safe(query_commit, max_gb_scanned=10)
display(num_commits)
   
    
import matplotlib.pyplot as plt
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','red']
patches,text=plt.pie(num_commits[["commit_count"]],colors=colors,shadow=True)
labels=num_commits["repo_name"]
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()
import seaborn as sns
sns.set_style("whitegrid")
ax = sns.barplot(x="commit_count", y="repo_name", data=num_commits)